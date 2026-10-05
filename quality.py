"""Conservative tail cleanup, verified near-duplicate removal and regrouped splits."""

import argparse
import collections
import contextlib
import functools
import gzip
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
import time

from datasketch import MinHash, MinHashLSH

import curate
import datawork as d
import newswork as nw

ROOT = Path(__file__).resolve().parent
DATASETS = ("common_pile_news", "cc_news", "hansard")


class Groups(d.UnionFind):
    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def normalized_line(text):
    return " ".join(text.split())


def tail_lines(text):
    return {normalized_line(line) for line in text.splitlines()[-14:] if 3 <= len(normalized_line(line)) <= 210}


def trim_tail(row, policy, frequencies):
    text, publisher = row["text"], row["publisher"]
    lines = [(match.start(), normalized_line(match.group())) for match in re.finditer(r"[^\n]+", text)]
    minimum = len(text) * policy["footer_minimum_document_fraction"]
    exact = set(policy["footer_lines"].get(publisher, []))
    allowed = exact | set(policy.get("additional_allowed_footer_tail_lines", {}).get(publisher, []))
    def recognized_footer(line):
        if line in allowed:
            return True
        if publisher in policy["related_snippet_publishers"]:
            return (line.rstrip(".").endswith("…") and len(line.split()) <= 50) or bool(re.fullmatch(r"Follow @\w+", line))
        return False
    candidates = [(offset, "source_specific_footer_line") for i, (offset, line) in enumerate(lines)
                  if offset >= minimum and line in exact and all(recognized_footer(x) for _, x in lines[i:])]
    if publisher in policy["related_snippet_publishers"]:
        snippets = [(offset, line) for offset, line in lines
                    if offset >= minimum and line.rstrip(".").endswith("…") and len(line.split()) <= 50
                    and frequencies[(publisher, line)] >= policy["recurring_related_snippet_minimum_documents"]]
        # A repeated multi-story tail is stronger evidence than one ellipsis.
        if (len(snippets) >= 3 and len(text[snippets[0][0]:].splitlines()) <= 14
                and all(recognized_footer(line) for offset, line in lines if offset >= snippets[0][0])):
            candidates.append((snippets[0][0], "recurring_related_story_tail"))
    if not candidates:
        return len(text), None
    offset, reason = min(candidates)
    return len(text[:offset].rstrip()), reason


def shingles(text, width=7):
    words = re.findall(r"\w+(?:['’]\w+)*", d.normalized(text))
    return {" ".join(words[i:i + width]) for i in range(len(words) - width + 1)}


def jaccard(first, second):
    union = len(first | second)
    return len(first & second) / union if union else 0.0


def greedy_keep(ordered_ids, adjacency):
    """Every removal needs a direct edge to an already retained document."""
    kept, removed = set(), {}
    for key in ordered_ids:
        retained_neighbors = sorted(kept.intersection(adjacency.get(key, set())))
        if retained_neighbors:
            removed[key] = retained_neighbors[0]
        else:
            kept.add(key)
    return kept, removed


def run(args):
    source, output, audit = args.input.resolve(), args.output.resolve(), args.audit.resolve()
    policy = json.loads(args.config.read_text())
    for directory in (output, audit):
        if directory == source or source in directory.parents or (directory.exists() and any(directory.iterdir())):
            raise ValueError("New output and audit directories must be empty and separate from inputs")
    if output == audit or output in audit.parents or audit in output.parents:
        raise ValueError("Audit and corpus outputs must be separate")
    original = json.loads((source / "manifest.json").read_text())
    input_manifest_sha = nw.digest_file(source / "manifest.json")
    verification = json.loads((source / "verification.json").read_text())
    if not verification["verified"] or verification["manifest_sha256"] != input_manifest_sha:
        raise ValueError("Source curation verification does not match")
    paths = [source / f"{name}.jsonl.gz" for name in DATASETS]
    for path in paths:
        if nw.digest_file(path) != original["outputs"][path.name]:
            raise ValueError(f"Retained input checksum mismatch: {path}")
    output.mkdir(parents=True, exist_ok=True); audit.mkdir(parents=True, exist_ok=True)
    nw.save_json(audit / "policy.json", policy)
    nw.save_json(output / "policy.json", policy)
    nw.save_json(output / "licenses.snapshot.json", json.loads((ROOT / "config/licenses.json").read_text()))
    frequencies, examples = collections.Counter(), {}
    for path in paths:
        for row in read_rows(path):
            for line in tail_lines(row["text"]):
                identity = row["publisher"], line
                frequencies[identity] += 1
                examples.setdefault(identity, row["id"])
    d.write_jsonl(audit / "recurring_tail_lines.jsonl", [
        {"publisher": publisher, "line": line, "document_frequency": count, "example_id": examples[(publisher, line)]}
        for (publisher, line), count in sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0])) if count >= 20])
    counts, reasons, by_publisher = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    groups, group_by_id, adjacency, pair_scores = Groups(), {}, collections.defaultdict(set), {}
    template = MinHash(num_perm=policy["minhash_permutations"], seed=policy["minhash_seed"], scheme=policy["minhash_scheme"])
    lsh = MinHashLSH(num_perm=policy["minhash_permutations"], params=(policy["lsh_bands"], policy["lsh_rows"]))
    with contextlib.ExitStack() as stack:
        database = stack.enter_context(contextlib.closing(sqlite3.connect(audit / "documents.sqlite")))
        database.execute("CREATE TABLE documents(id TEXT PRIMARY KEY, dataset TEXT, words INT, shingle_count INT, license_rank INT, record TEXT)")
        database.execute("CREATE TABLE spans(id TEXT PRIMARY KEY, parent TEXT, start INT, end INT, words INT, digest TEXT)")
        database.execute("CREATE INDEX spans_by_parent ON spans(parent,start)")
        changes = stack.enter_context(gzip.open(audit / "tail_changes.jsonl.gz", "wt", encoding="utf-8"))
        rejected = stack.enter_context(gzip.open(output / "rejections.jsonl.gz", "wt", encoding="utf-8"))
        pairs = stack.enter_context(gzip.open(audit / "similarity_pairs.jsonl.gz", "wt", encoding="utf-8"))

        @functools.lru_cache(maxsize=96)
        def prior_shingles(key):
            row = json.loads(database.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()[0])
            return shingles(row["text"], policy["shingle_words"])

        last_progress = time.monotonic()
        for path in paths:
            for row in read_rows(path):
                counts["input_documents"] += 1
                key = row["id"]
                end, trim_reason = trim_tail(row, policy, frequencies)
                original_text, original_hash = row["text"], row["text_sha256"]
                text = original_text[:end]
                words = len(text.split())
                if trim_reason:
                    counts["tail_trimmed_documents"] += 1
                    counts["tail_removed_words"] += row["whitespace_words"] - words
                    by_publisher[row["publisher"]]["trimmed_documents"] += 1
                    by_publisher[row["publisher"]]["removed_words"] += row["whitespace_words"] - words
                    changes.write(json.dumps({"id": key, "dataset": row["dataset"], "publisher": row["publisher"], "source_text_sha256": original_hash,
                                              "retained_character_range": [0, end], "reason": trim_reason,
                                              "removed_words": row["whitespace_words"] - words, "removed_tail": original_text[end:]}, ensure_ascii=False) + "\n")
                reason = "body_below_minimum_after_tail_cleanup" if words < policy["minimum_words"] else None
                if reason is None and curate.MT_NOTICE.search(text):
                    reason = "possible_explicit_mt_disclosure_requires_review"
                if reason:
                    reasons[f"{row['dataset']}:{reason}"] += 1
                    rejected.write(json.dumps({"id": key, "dataset": row["dataset"], "reason": reason, "source_text_sha256": original_hash}) + "\n")
                    continue
                row.update(text=text, text_sha256=d.text_hash(text), whitespace_words=words,
                           quality_v2={"source_record_id": key, "source_text_sha256": original_hash,
                                       "source_character_range": [0, end], "tail_rule": trim_reason,
                                       "source_curation_manifest_sha256": input_manifest_sha})
                terms = shingles(text, policy["shingle_words"])
                mh = template.copy(); mh.update_batch(term.encode("utf-8") for term in terms)
                group_by_id[key] = row["split_group"]
                groups.find(row["split_group"])
                for other in sorted(lsh.query(mh)):
                    counts["lsh_candidate_pairs"] += 1
                    size, raw = database.execute("SELECT shingle_count,record FROM documents WHERE id=?", (other,)).fetchone()
                    if min(size, len(terms)) / max(size, len(terms)) < policy["group_jaccard_minimum"]:
                        counts["candidate_pairs_below_size_upper_bound"] += 1
                        continue
                    similarity = jaccard(terms, prior_shingles(other))
                    counts["exact_shingle_comparisons"] += 1
                    if similarity < policy["group_jaccard_minimum"]:
                        continue
                    prior = json.loads(raw)
                    ratio = min(words, prior["whitespace_words"]) / max(words, prior["whitespace_words"])
                    duplicate = similarity >= policy["dedup_jaccard_minimum"] and ratio >= policy["dedup_word_ratio_minimum"]
                    groups.union(row["split_group"], prior["split_group"])
                    counts["confirmed_similarity_pairs"] += 1
                    counts["confirmed_pairs_crossing_v1_splits"] += row["split"] != prior["split"]
                    if duplicate:
                        adjacency[key].add(other); adjacency[other].add(key)
                        pair_scores[tuple(sorted((key, other)))] = similarity
                        counts["confirmed_dedup_pairs"] += 1
                    pairs.write(json.dumps({"first": other, "second": key, "shingle_jaccard": similarity, "word_ratio": ratio,
                                            "dedup_eligible": duplicate, "source_splits": [prior["split"], row["split"]]}) + "\n")
                database.execute("INSERT INTO documents VALUES(?,?,?,?,?,?)", (key, row["dataset"], words, len(terms), int(row["license"] == "unknown"), json.dumps(row, ensure_ascii=False)))
                lsh.insert(key, mh)
                counts["eligible_after_cleanup"] += 1
                if time.monotonic() - last_progress > 25:
                    database.commit()
                    print(f"Quality scan: {counts['input_documents']:,} articles, {counts['confirmed_similarity_pairs']:,} confirmed pairs", flush=True)
                    last_progress = time.monotonic()
            database.commit()
        ordered = [row[0] for row in database.execute("SELECT id FROM documents ORDER BY license_rank,words DESC,id")]
        kept, removed = greedy_keep(ordered, adjacency)
        d.write_jsonl(audit / "near_duplicate_decisions.jsonl", [
            {"removed_id": key, "retained_id": retained, "shingle_jaccard": pair_scores[tuple(sorted((key, retained)))],
             "reason": "direct_verified_near_duplicate_of_preferred_retained_copy"} for key, retained in sorted(removed.items())])
        counts["near_duplicate_documents_removed"] = len(removed)
        for key in sorted(kept):
            row = json.loads(database.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()[0])
            for start, end, words in curate.make_spans(row["text"], policy["minimum_words"], policy["maximum_span_words"]):
                text = row["text"][start:end]
                database.execute("INSERT INTO spans VALUES(?,?,?,?,?,?)", (f"span/{key}/{start}-{end}", key, start, end, words, d.text_hash(d.normalized(text))))
        database.commit()
        # Exact partial passages can recur even when whole articles differ.
        span_groups = {}
        for parent, digest in database.execute("SELECT parent,digest FROM spans ORDER BY id"):
            group = group_by_id[parent]
            if digest in span_groups:
                groups.union(group, span_groups[digest])
                counts["repeated_exact_span_identities"] += 1
            else:
                span_groups[digest] = group
        documents = {name: stack.enter_context(gzip.open(output / f"{name}.jsonl.gz", "wt", encoding="utf-8")) for name in DATASETS}
        spans = {name: stack.enter_context(gzip.open(output / f"{name}.spans.jsonl.gz", "wt", encoding="utf-8")) for name in DATASETS}
        coverage, sample = collections.defaultdict(collections.Counter), collections.defaultdict(list)
        final_partitions = {}
        for key in sorted(kept):
            row = json.loads(database.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()[0])
            group = groups.find(group_by_id[key])
            bucket = int(d.text_hash(policy["split_seed"] + ":" + group)[:16], 16) % 100
            split = "train" if bucket < 98 else "validation" if bucket == 98 else "test"
            counts["retained_documents_reassigned_split"] += row["split"] != split
            row["quality_v2"]["source_split"] = row["split"]
            row.update(split_group=group, split=split)
            final_partitions[key] = split
            dataset, publisher = row["dataset"], row["publisher"]
            documents[dataset].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[f"retained_documents:{dataset}"] += 1
            counts[f"retained_words:{dataset}"] += row["whitespace_words"]
            stats = coverage[f"{dataset}:{publisher}"]
            stats["documents"] += 1; stats["words"] += row["whitespace_words"]
            stats[f"split:{split}"] += 1
            for span_id, start, end, words in database.execute("SELECT id,start,end,words FROM spans WHERE parent=? ORDER BY start", (key,)):
                text = row["text"][start:end]
                span = {**row, "id": span_id, "parent_document_id": key, "text": text, "text_sha256": d.text_hash(text),
                        "whitespace_words": words, "character_range": [start, end], "original_character_range": [start, end]}
                spans[dataset].write(json.dumps(span, ensure_ascii=False) + "\n")
                counts[f"retained_spans:{dataset}"] += 1; counts[f"retained_span_words:{dataset}"] += words
                stats["spans"] += 1; stats["span_words"] += words
            review = sample[(dataset, publisher)]
            review.append((d.text_hash("quality-v2-review:" + key), row)); review.sort(key=lambda x: x[0]); del review[8:]
        d.write_jsonl(output / "review_queue.jsonl", [
            {**row, "reviewer": None, "confirmed_human_evidence": None, "confirmed_version_evidence": None, "review_decision": None}
            for stratum, selected in sorted(sample.items()) for _, row in selected])
        nw.save_json(audit / "cleanup_by_publisher.json", {k: dict(v) for k, v in by_publisher.items()})
    # Export indexes for parent lookups in verification without copying all raw text.
    report = {"processing_complete": True, "input_curation": str(source), "input_manifest_sha256": nw.digest_file(source / "manifest.json"),
              "input_files": {str(p): nw.digest_file(p) for p in paths}, "counts": dict(counts), "rejections": dict(reasons),
              "sources": {k: dict(v) for k, v in coverage.items()}, "audit_directory": str(audit),
              "config_sha256": nw.digest_file(args.config), "script_sha256": nw.digest_file(Path(__file__)),
              "span_script_sha256": nw.digest_file(Path(curate.__file__)), "datasketch_version": importlib.metadata.version("datasketch"),
              "lsh_retrieval_probability_at_group_threshold": 1 - (1 - policy["group_jaccard_minimum"] ** policy["lsh_rows"]) ** policy["lsh_bands"],
              "limitations": ["Candidate labels and authorship/version uncertainty unchanged",
                              "Exact Jaccard verification applies only to retrieved pairs; approximate LSH can miss duplicates",
                              "One retained copy per direct near-duplicate decision; connected groups are used for splits, not blanket deletion",
                              "Source-specific footer rules do not remove all site boilerplate, comments or recent article edits",
                              "No learned detector applied; remaining synthetic quotations and undisclosed MT not ruled out",
                              "Splits regrouped from v1; do not mix v1 partitions with v2 for evaluation"]}
    report["outputs"] = {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    report["audit_outputs"] = {p.name: nw.digest_file(p) for p in sorted(audit.iterdir()) if p.is_file()}
    nw.save_json(output / "manifest.json", report)
    nw.save_json(audit / "manifest.json", {"output_curation": str(output), "output_manifest_sha256": nw.digest_file(output / "manifest.json"), "counts": dict(counts), "outputs": report["audit_outputs"]})
    print(json.dumps({"counts": dict(counts), "rejections": dict(reasons)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/curated-v1")
    parser.add_argument("--config", type=Path, default=ROOT / "config/quality.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/curated-v2")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/quality-v2")
    run(parser.parse_args())
