"""Build a genre-balanced, provenance-screened English human-origin candidate mixture."""

import argparse
import collections
import copy
import gzip
import json
from pathlib import Path
import re
from urllib.parse import quote

from datasketch import MinHash, MinHashLSH
from lingua import Language, LanguageDetectorBuilder
import pyarrow.parquet as pq

import curate
import datawork as d
import newswork as nw
import quality
import strictsubset as strict
import wordcorpus

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/diverse-human"
POLICY = ROOT / "config/diverse_human.json"
NOTICE = re.compile(strict.NOTICE.pattern + r"|\b(?:automated insights|narrative science|wordsmith|heliograf|robot[ -]written|AI[ -]generated)\b", re.I)
TITLE = re.compile(r"^\s*=\s+([^=\n]+?)\s+=\s*$")


def named_person(value):
    if not isinstance(value, str) or re.search(r"\b(?:staff|editor|admin|team|news|press|reuters|agency)\b", value, re.I):
        return False
    tokens = value.strip().split()
    return 2 <= len(tokens) <= 10 and sum(t[0].isupper() for t in tokens if t) >= 2 and not any(c.isdigit() for c in value)


def byline(row):
    author = row.get("metadata", {}).get("author")
    if named_person(author) and d.normalized(author) in d.normalized(row["text"][:1600]):
        return author.strip(), "Named metadata author also present near article beginning"
    for pattern, sample in [(r"\b(?:Reporting|Writing) by ([^;\n)]+)", row["text"][-1800:]), (r"(?m)^By ([^\n|]+)", row["text"][:1400])]:
        match = re.search(pattern, sample)
        if match and named_person(match.group(1)):
            return match.group(1).strip(), "Explicit reporting/writing credit in article text"
    return None, None


def restore_punctuation(text):
    # Formatting only: retain the released words; no language model or paraphrase.
    text = re.sub(r"\s+@-@\s+", "-", text)
    text = re.sub(r"\s+@,@\s+", ",", text)
    text = re.sub(r"\s+@\.@\s+", ".", text)
    text = re.sub(r"[ \t]+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])[ \t]+", r"\1", text)
    return re.sub(r"(?<=\w)[ \t]+('[sdm]|'re|'ve|'ll|n't)\b", r"\1", text)


def book_body(text):
    start = re.search(r"(?m)^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\r\n]*\*\*\*\r?\n", text)
    end = re.search(r"(?m)^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\r\n]*\*\*\*", text)
    if not start or not end or start.end() >= end.start():
        raise ValueError("Missing or reversed Gutenberg body markers")
    return start.end(), end.start()


def wiki_articles(texts):
    title, begin, parts = None, None, []
    for index, text in enumerate(texts):
        match = TITLE.fullmatch(text)
        # Genuine article headings are separate blank-bounded blocks. Inline
        # equations/table keys also resemble single-equals headings otherwise.
        bounded = ((index == 0 or not texts[index - 1].strip())
                   and (index + 1 == len(texts) or not texts[index + 1].strip()))
        if match and bounded:
            if title is not None:
                yield title, begin, index, "".join(parts)
            title, begin, parts = match.group(1).strip(), index, [text]
        elif title is not None:
            parts.append(text)
        elif text.strip():
            raise ValueError("WikiText body appears before first article title")
    if title is not None:
        yield title, begin, len(texts), "".join(parts)


def news_gate(row, policy):
    hosts = policy["common_pile_publishers"] if row["dataset"] == "common_pile_news" else policy["cc_publishers"]
    if row["publisher"] not in hosts or row["publication_date"] > policy["news_publication_cutoff"]:
        return "publisher_or_date_outside_news_policy", None
    if row.get("last_known_update_date") and row["last_known_update_date"] > policy["news_known_update_cutoff"]:
        return "known_news_update_after_cutoff", None
    if NOTICE.search(row["text"]) or strict.PARSER_NOTE.search(row["text"]):
        return "generation_translation_or_extraction_mention_requires_review", None
    author, basis = byline(row)
    if not author:
        return "missing_named_article_byline", None
    return None, {"author": author, "basis": basis}


def prose_ok(text, policy):
    metrics = strict.prose_metrics(text)
    return (metrics["alphabetic_token_fraction"] >= policy["minimum_alphabetic_token_fraction"]
            and metrics["digit_token_fraction"] <= policy["maximum_digit_token_fraction"]
            and metrics["numeric_line_fraction"] <= policy["maximum_numeric_line_fraction"])


def load_pool(policy, inventory_path=None):
    inventory_path = inventory_path or ROOT / "config/licenses.json"
    docs, spans, exclusions, input_hashes = {}, [], collections.Counter(), {str(inventory_path): nw.digest_file(inventory_path)}
    corpus, parent_dir, core = ROOT / "data/corpus-words-v2", ROOT / "data/curated-v2", ROOT / "data/corpus-human-core-v1"
    _, checked = strict.checked_inputs(corpus, parent_dir)
    input_hashes.update(checked)
    for directory in (core,):
        manifest = strict.read(directory / "manifest.json")
        report = strict.read(directory / "verification.json")
        if not report["verified"] or report["manifest_sha256"] != nw.digest_file(directory / "manifest.json"):
            raise ValueError("Strict core not verified")
        for split in strict.SPLITS:
            path = directory / (split + ".jsonl.gz")
            if nw.digest_file(path) != manifest["outputs"][path.name]:
                raise ValueError("Core file changed")
            input_hashes[str(path)] = nw.digest_file(path)
    parent_manifest = strict.read(parent_dir / "manifest.json")
    inventory = strict.read(inventory_path)
    for dataset in ("common_pile_news", "cc_news", "hansard"):
        path = parent_dir / (dataset + ".jsonl.gz")
        if nw.digest_file(path) != parent_manifest["outputs"][path.name]:
            raise ValueError("Parent archive changed")
        input_hashes[str(path)] = nw.digest_file(path)
        for row in strict.rows(path):
            if dataset == "hansard":
                docs[row["id"]] = row
                continue
            reason, evidence = news_gate(row, policy)
            if reason:
                exclusions[reason] += 1
                continue
            row["author"] = evidence["author"]
            row["genre"] = "journalism"
            row["human_origin_basis"] = {**evidence, "kind": "named_pre_2019_article", "individual_authorship_verified": False,
                                         "captured_version_verified": False, "source_policy_is_not_retroactive_attestation": True}
            wordcorpus.annotate_license(row, inventory)
            docs[row["id"]] = row
    for split in strict.SPLITS:
        for row in strict.rows(core / (split + ".jsonl.gz")):
            parent = docs[row["parent_document_id"]]
            parent.update(genre="parliament", author="Named parliamentary participants", human_origin_basis=row["human_core_selection"])
            spans.append({**row, "genre": "parliament", "author": parent["author"], "human_origin_basis": parent["human_origin_basis"]})
        for row in strict.rows(corpus / (split + ".jsonl.gz")):
            parent = docs.get(row["parent_document_id"])
            if parent and parent.get("genre") == "journalism" and prose_ok(row["text"], policy):
                spans.append({**row, "genre": "journalism", "author": parent["author"], "human_origin_basis": parent["human_origin_basis"]})
    acquisition = strict.read(RAW / "acquisition.json")
    books = [x for x in acquisition["files"] if x["dataset"] == "gutenberg_selected"]
    author_order = sorted({x["author"] for x in books}, key=lambda a: d.text_hash(policy["seed"] + a))
    book_splits = {author: ("test" if i == 0 else "validation" if i == 1 else "train") for i, author in enumerate(author_order)}
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    for spec in acquisition["files"]:
        path = RAW / spec["local_path"]
        if path.stat().st_size != spec["bytes"] or nw.digest_file(path) != spec["sha256"]:
            raise ValueError("Acquired raw file changed")
        input_hashes[str(path)] = spec["sha256"]
        if spec["dataset"] == "gutenberg_selected":
            raw_text = path.read_bytes().decode("utf-8-sig")
            start, end = book_body(raw_text)
            documents = [(f"gutenberg/{spec['book_id']}", raw_text[start:end], {
                "dataset": "gutenberg_selected", "genre": spec["genre"], "author": spec["author"], "title": spec["title"],
                "publisher": "Project Gutenberg", "url": spec["provenance_url"], "source": "historical_original_english_work",
                "original_publication_year": spec["original_publication_year"], "publication_date": None,
                "split": book_splits[spec["author"]], "split_group": "book-author/" + spec["author"],
                "body_character_range": [start, end], "license": "Public domain in USA declared by Project Gutenberg; ebook/trademark terms and jurisdiction-specific rights retained",
                "license_status": "source_declared_public_domain_not_global_rights_certification",
                "human_origin_basis": {"kind": "identified_historical_human_authored_work", "original_language": "en", "acquired_ebook_revision_date_unknown": True},
                "source_file": str(path), "source_file_sha256": spec["sha256"]})]
        elif path.suffix == ".parquet":
            texts = pq.read_table(path)["text"].to_pylist()
            documents = [("wikitext/" + title, restore_punctuation(text), {
                "dataset": "wikitext2_raw", "genre": "reference", "author": "Wikipedia contributors", "title": title,
                "publisher": "Wikipedia", "url": "https://en.wikipedia.org/wiki/" + quote(restore_punctuation(title).replace(" ", "_")),
                "url_evidence": "Inferred from released article title with mechanical punctuation restoration; revision ID unavailable",
                "source": "2016_wikitext2_raw_release", "publication_date": None, "corpus_release_year": 2016,
                "split": spec["upstream_split"], "split_group": "wikitext/" + title,
                "source_row_range": [begin, end], "raw_article_text_sha256": d.text_hash(text),
                "formatting": "Deterministic punctuation-marker/spacing restoration; no generated wording",
                "license": "CC-BY-SA/GFDL; pinned card header says 3.0 and body says 4.0; version reconciliation required",
                "license_status": "declared_license_version_discrepancy_requires_review",
                "human_origin_basis": {"kind": "pre_LLM_good_and_featured_Wikipedia_release", "individual_authorship_verified": False, "article_revision_ids_unavailable": True},
                "source_file": str(path), "source_file_sha256": spec["sha256"]})
                for title, begin, end, text in wiki_articles(texts)]
        else:
            continue
        for key, text, metadata in documents:
            if key in docs:
                raise ValueError("Repeated new document identity; regroup before importing")
            if NOTICE.search(text) or strict.PARSER_NOTE.search(text):
                exclusions[metadata["dataset"] + ":generation_or_extraction_mention_requires_review"] += 1
                continue
            scores = [detector.compute_language_confidence(sample, Language.ENGLISH) for sample in curate.language_samples(text)]
            if min(scores) < 0.8:
                exclusions[metadata["dataset"] + ":english_confidence_below_minimum"] += 1
                continue
            parent = {**metadata, "id": key, "text": text, "text_sha256": d.text_hash(text), "whitespace_words": len(text.split()),
                      "language": "en", "authorship": "human_candidate", "binary_target": None, "english_sample_confidences": scores,
                      "flags": ["individual_authorship_unverified"], "review_status": "source_provenance_selected"}
            docs[key] = parent
            for a, b, words in curate.make_spans(text, policy["minimum_words"], policy["maximum_words"]):
                if prose_ok(text[a:b], policy):
                    spans.append({**parent, "id": f"span/{key}/{a}-{b}", "text": text[a:b], "text_sha256": d.text_hash(text[a:b]),
                                  "whitespace_words": words, "parent_document_id": key, "character_range": [a, b]})
    return docs, spans, exclusions, input_hashes


def balanced_select(spans, policy):
    capacities = collections.Counter()
    for row in spans:
        capacities[row["genre"]] += row["whitespace_words"]
    total = int(min(capacities[g] / weight for g, weight in policy["genre_weights"].items()))
    budgets = {g: int(total * w) for g, w in policy["genre_weights"].items()}
    selected, used, publishers, authors, parents, digests = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), set()
    for row in sorted(spans, key=lambda r: d.text_hash(policy["seed"] + r["id"])):
        genre, words = row["genre"], row["whitespace_words"]
        if used[genre] + words > budgets[genre]:
            continue
        if genre == "journalism" and publishers[row["publisher"]] + words > total * policy["publisher_maximum_total_fraction"]:
            continue
        if genre in {"fiction", "nonfiction"} and authors[row["author"]] + words > total * policy["author_maximum_total_fraction"]:
            continue
        if genre in {"journalism", "reference", "parliament"} and parents[row["parent_document_id"]] + words > total * policy["document_maximum_total_fraction"]:
            continue
        digest = d.text_hash(d.normalized(row["text"]))
        if digest in digests:
            continue
        digests.add(digest)
        selected.append(copy.deepcopy(row))
        used[genre] += words
        publishers[row["publisher"]] += words
        authors[row["author"]] += words
        parents[row["parent_document_id"]] += words
    return selected, {"eligible_genre_words": dict(capacities), "capacity_balanced_target_words": total, "genre_budgets": budgets, "selected_genre_words": dict(used)}


def group_similar(selected):
    groups, pairs, stored = quality.Groups(), [], {}
    template = MinHash(num_perm=128, seed=42, scheme="affine64")
    lsh = MinHashLSH(num_perm=128, params=(16, 8))
    for row in selected:
        terms = quality.shingles(row["text"])
        mh = template.copy(); mh.update_batch(x.encode() for x in terms)
        groups.find(row["split_group"])
        for key in sorted(lsh.query(mh)):
            prior, other_terms = stored[key]
            if min(len(terms), len(other_terms)) / max(len(terms), len(other_terms), 1) < 0.8:
                continue
            similarity = quality.jaccard(terms, other_terms)
            if similarity >= 0.8:
                groups.union(row["split_group"], prior["split_group"])
                pairs.append({"first": row["id"], "second": prior["id"], "jaccard": similarity})
        lsh.insert(row["id"], mh); stored[row["id"]] = row, terms
    partitions = {}
    order = {"train": 0, "validation": 1, "test": 2}
    for row in selected:
        group = groups.find(row["split_group"])
        prior = partitions.get(group, row["split"])
        partitions[group] = max(prior, row["split"], key=order.get)
    for row in selected:
        row["upstream_split"] = row["split"]
        row["upstream_split_group"] = row["split_group"]
        row["split_group"] = groups.find(row["split_group"])
        row["split"] = partitions[row["split_group"]]
    return pairs


def write_rows(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(selected):
    report = {k: collections.defaultdict(collections.Counter) for k in ("by_genre", "by_dataset", "by_publisher", "by_author", "by_split")}
    for row in selected:
        for key, field in [("by_genre", "genre"), ("by_dataset", "dataset"), ("by_publisher", "publisher"), ("by_author", "author"), ("by_split", "split")]:
            report[key][row[field]]["passages"] += 1
            report[key][row[field]]["words"] += row["whitespace_words"]
    return {k: {a: dict(b) for a, b in v.items()} for k, v in report.items()}


def build(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    policy = strict.read(POLICY)
    docs, spans, excluded, inputs = load_pool(policy)
    selected, allocation = balanced_select(spans, policy)
    pairs = group_similar(selected)
    output.mkdir(parents=True, exist_ok=True)
    for split in strict.SPLITS:
        write_rows(output / (split + ".jsonl.gz"), (r for r in selected if r["split"] == split))
    parent_ids = sorted({r["parent_document_id"] for r in selected})
    write_rows(output / "parent_documents.jsonl.gz", (docs[key] for key in parent_ids))
    nw.save_json(output / "similarity_pairs.json", pairs)
    nw.save_json(output / "policy.json", policy)
    nw.save_json(output / "licenses.snapshot.json", strict.read(ROOT / "config/licenses.json"))
    summary = summarize(selected)
    nw.save_json(output / "diversity.json", {**summary, **allocation})
    (output / "ATTRIBUTION.txt").write_text("Contains Parliamentary information licensed under the Open Parliament Licence v3.0.\nWikipedia contributors via WikiText-2; per-article source URLs retained; CC BY-SA/GFDL version discrepancy recorded.\nProject Gutenberg titles/authors/source URLs retained; ebook license and trademark terms are in raw files.\nNews source URLs/bylines and original license fields retained; unknown/disputed rights remain in inventory.\n")
    manifest = {"processing_complete": True, "script_sha256": nw.digest_file(Path(__file__)), "policy_sha256": nw.digest_file(POLICY),
                "input_files": inputs, "acquisition_sha256": nw.digest_file(RAW / "acquisition.json"),
                "passages": len(selected), "words": sum(r["whitespace_words"] for r in selected), "parent_documents": len(parent_ids),
                "exclusions": dict(excluded), "summary": summary, "allocation": allocation, "detected_similar_pairs": len(pairs),
                "split_changes": sum(r["split"] != r["upstream_split"] for r in selected),
                "confidence": "Provenance-screened human-origin candidates across five genres; source confidence bases differ and purity is not measured",
                "limitations": policy["notes"] + ["Approximate passage-level LSH can miss pairs; similarity grouping is not comprehensive duplicate certification", "Inherited benchmark formatting and domain/era differences can become detector shortcuts"],
                "outputs": {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
    nw.save_json(output / "manifest.json", manifest)
    print(json.dumps({"passages": manifest["passages"], "words": manifest["words"], "parents": len(parent_ids), "genres": summary["by_genre"], "publishers": summary["by_publisher"]}, indent=2), flush=True)


def verify(output):
    manifest = strict.read(output / "manifest.json")
    for name, expected in manifest["outputs"].items():
        if Path(name).name != name or nw.digest_file(output / name) != expected:
            raise ValueError("Output checksum mismatch")
    for name, expected in manifest["input_files"].items():
        path = output / "licenses.snapshot.json" if Path(name) == ROOT / "config/licenses.json" else Path(name)
        if nw.digest_file(path) != expected:
            raise ValueError("Input checksum mismatch")
    if nw.digest_file(RAW / "acquisition.json") != manifest["acquisition_sha256"]:
        raise ValueError("Acquisition pin changed")
    docs, candidates, _, _ = load_pool(strict.read(output / "policy.json"), output / "licenses.snapshot.json")
    expected, _ = balanced_select(candidates, strict.read(output / "policy.json"))
    group_similar(expected)
    actual = [r for split in strict.SPLITS for r in strict.rows(output / (split + ".jsonl.gz"))]
    if {r["id"]: r for r in expected} != {r["id"]: r for r in actual} or len(actual) != len(expected):
        raise ValueError("Selection differs from reproducible source derivation")
    groups, texts = {}, set()
    for row in actual:
        parent = docs[row["parent_document_id"]]
        a, b = row["character_range"]
        words = len(row["text"].split())
        if parent["text"][a:b] != row["text"] or d.text_hash(row["text"]) != row["text_sha256"] or words != row["whitespace_words"] or not 400 <= words <= 3000:
            raise ValueError("Source slice, count or text mismatch")
        if row["binary_target"] is not None or row["authorship"] != "human_candidate" or row["language"] != "en":
            raise ValueError("Candidate scope mismatch")
        digest = d.text_hash(d.normalized(row["text"]))
        if digest in texts or groups.setdefault(row["split_group"], row["split"]) != row["split"]:
            raise ValueError("Duplicate text or cross-partition source group")
        texts.add(digest)
    parent_export = {r["id"]: r for r in strict.rows(output / "parent_documents.jsonl.gz")}
    if parent_export != {r["parent_document_id"]: docs[r["parent_document_id"]] for r in actual}:
        raise ValueError("Parent provenance export differs")
    if summarize(actual) != manifest["summary"] or len(actual) != manifest["passages"] or sum(r["whitespace_words"] for r in actual) != manifest["words"]:
        raise ValueError("Diversity/count report differs")
    nw.save_json(output / "verification.json", {"verified": True, "manifest_sha256": nw.digest_file(output / "manifest.json"),
                 "verifier_sha256": nw.digest_file(Path(__file__)), "passages": len(actual), "words": manifest["words"],
                 "checks": ["Pinned inputs and output hashes", "Reproducible genre allocation", "Source bodies and exact contiguous slices", "Word bounds", "Unique text and parent partitions", "Recorded author/source limits", "Cross-source retrieved similarity groups"],
                 "scope": "Integrity, allocation and recorded provenance; no measured human-only guarantee"})
    print("Verified", len(actual), "passages", manifest["words"], "words", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/corpus-human-diverse-v1")
    args = parser.parse_args()
    if args.command == "build":
        build(args.output.resolve())
    else:
        verify(args.output.resolve())
