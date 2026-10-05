"""Curate local news and Hansard into unbudgeted, auditable human-origin candidates."""

import argparse
import bisect
import collections
import contextlib
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pyarrow.parquet as pq
from lingua import Language, LanguageDetectorBuilder

import datawork as d
import newswork as nw

ROOT = Path(__file__).resolve().parent
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
DATE_RE = re.compile(rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}[,]?\s+\d{{4}})\b", re.I)
UPDATE_RE = re.compile(r"(?:last\s+)?(?:updated|modified)(?:\s+on)?\s*:?[\s(]*", re.I)
MT_NOTICE = re.compile(r"\bthis\s+(?:article|story|page|text)\s+(?:has\s+been|was|is)\s+(?:automatically|machine)[ -]translated\b|\btranslated\s+(?:using|with|by)\s+(?:google\s+translate|deepl)\b", re.I)


def explicit_dates(value):
    dates = []
    for match in DATE_RE.finditer(value or ""):
        candidate = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", match.group(), flags=re.I).replace(",", "")
        formats = ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y")
        for fmt in formats:
            try:
                date = dt.datetime.strptime(candidate, fmt).date()
                dates.append(date)
                break
            except ValueError:
                pass
    return dates


def news_dates(row, dataset, config):
    if dataset == "cc_news":
        try:
            date = dt.datetime.fromisoformat(row.get("date", "")).date()
            return date, None, "Upstream extracted publication-date field; capture timestamp absent", None
        except (ValueError, TypeError):
            return None, None, None, "missing_or_invalid_publication_date"
    created = row.get("created") or ""
    marker = UPDATE_RE.search(created)
    published = created[:marker.start()] if marker else created
    dates = sorted(set(explicit_dates(published)))
    if len(dates) > 1:
        return None, None, None, "ambiguous_publication_dates"
    url = (row.get("metadata") or {}).get("url") or ""
    url_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", urlsplit(url).path)
    url_date = None
    if url_match and row.get("source") in config["date_from_daily_url_sources"]:
        try:
            url_date = dt.date(*(int(part) for part in url_match.groups()))
        except ValueError:
            return None, None, None, "invalid_publication_date_in_url"
    if dates and url_date and dates[0] != url_date:
        return None, None, None, "publication_date_url_conflict"
    date = dates[0] if dates else url_date
    evidence = "Parsed source publication field" if dates else "Source-specific dated article URL; revision timestamp absent"
    if date is None:
        return None, None, None, "missing_or_unparsed_publication_date"
    updates = explicit_dates(created[marker.end():]) if marker else []
    if marker and not updates:
        return None, None, None, "unparsed_explicit_update_date"
    updated = max(updates) if updates else None
    if updated and updated < date:
        return None, None, None, "update_precedes_publication"
    return date, updated, evidence, None


def host(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Invalid article URL")
    return parsed.hostname.lower().removeprefix("www.")


def host_matches(actual, expected):
    return actual == expected or actual.endswith("." + expected)


def canonical_url(url):
    parsed = urlsplit(url)
    publisher = host(url)
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    return urlunsplit(("https", publisher, parsed.path.rstrip("/") or "/", urlencode(sorted(query)), ""))


def preliminary(row, dataset, config):
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "empty_text"
    words = len(text.split())
    if words < config["minimum_words"]:
        return None, "document_too_short_in_words"
    metadata = row.get("metadata") or {}
    url = row.get("url") or metadata.get("url")
    try:
        actual_host = host(url or "")
        canon = canonical_url(url)
    except ValueError:
        return None, "invalid_article_url"
    if dataset == "common_pile_news":
        expected = config["common_pile_source_hosts"].get(row.get("source"))
        if expected is None or not host_matches(actual_host, expected):
            return None, "source_host_mismatch_or_unreviewed_source"
        license = metadata.get("license") or "unknown"
        if not any(part in license for part in ("creativecommons.org/licenses/by/4.0/", "creativecommons.org/licenses/by-sa/4.0/")):
            return None, "unexpected_or_missing_declared_license"
        publisher = expected
    else:
        publisher = next((p for p in config["cc_news_publishers"] if host_matches(actual_host, p)), None)
        if publisher is None:
            return None, "publisher_requires_review"
        try:
            if host("https://" + row.get("domain", "")) != actual_host:
                return None, "domain_url_mismatch"
        except ValueError:
            return None, "missing_or_invalid_domain"
        license = "unknown"
    if urlsplit(canon).path == "/":
        return None, "homepage_not_article"
    date, updated, evidence, reason = news_dates(row, dataset, config)
    if reason:
        return None, reason
    cutoff = dt.date.fromisoformat(config["publication_cutoff"])
    if date > cutoff:
        return None, "publication_after_cutoff"
    if updated and updated > cutoff:
        return None, "explicit_update_after_cutoff"
    if MT_NOTICE.search(text[:1500] + "\n" + text[-1000:]):
        return None, "explicit_machine_translation_notice_requires_review"
    return {"publication_date": date.isoformat(), "last_known_update_date": updated.isoformat() if updated else None,
            "publication_date_evidence": evidence, "url": url, "canonical_url": canon,
            "publisher": publisher, "license": license, "whitespace_words": words}, None


def language_samples(text):
    if len(text) <= 1800:
        return [text]
    middle = max(0, len(text) // 2 - 900)
    return [text[:1800], text[middle:middle + 1800], text[-1800:]]


def make_spans(text, minimum, maximum):
    """Contiguous exact character slices; prefer paragraph or sentence endings."""
    if not 0 < minimum <= maximum:
        raise ValueError("Require 0 < minimum <= maximum span words")
    words = list(re.finditer(r"\S+", text))
    if len(words) <= maximum:
        return [(0, len(text), len(words))] if len(words) >= minimum else []
    starts = [word.start() for word in words]
    boundaries = [match.end() for match in re.finditer(r"\n\s*\n|[.!?][\"’”']?\s+", text)]
    spans, position = [], 0
    while position < len(text):
        word_start = bisect.bisect_left(starts, position)
        if len(starts) - word_start <= maximum:
            count = len(text[position:].split())
            if count >= minimum:
                spans.append((position, len(text), count))
            break
        limit_index = min(word_start + maximum, len(starts))
        end = len(text) if limit_index == len(starts) else starts[limit_index]
        left = bisect.bisect_right(boundaries, position)
        right = bisect.bisect_right(boundaries, end)
        if right > left and boundaries[right - 1] >= position + (end - position) * 0.65:
            preferred = boundaries[right - 1]
            if len(text[position:preferred].split()) >= minimum:
                end = preferred
        if end <= position:
            end = min(len(text), position + 1)
        count = len(text[position:end].split())
        if count >= minimum:
            spans.append((position, end, count))
        position = end
    return spans


def iter_file(path, dataset):
    if path.suffix == ".parquet":
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                index += 1
                yield index, row
    else:
        with gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, 1):
                yield index, json.loads(line)


def curate(args):
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing results will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    nw.save_json(output / "policy.json", config)
    nw.save_json(output / "licenses.snapshot.json", json.loads((ROOT / "config/licenses.json").read_text()))
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    downloads = json.loads(nw.CONFIG.read_text())
    inputs = [(dataset["name"], nw.RAW / spec["local_path"], spec["sha256"])
              for dataset in downloads["datasets"] for spec in dataset["files"]]
    prefix = ROOT / "data/raw/hansard_prefix.jsonl"
    prefix_meta = json.loads(prefix.with_suffix(".manifest.json").read_text())
    inputs.append(("hansard", prefix, prefix_meta["sha256"]))
    counts, rejects, by_source = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    groups = d.UnionFind()
    identities = {}
    manifests = []
    with contextlib.ExitStack() as stack:
        database = stack.enter_context(sqlite3.connect(output / "inventory.sqlite"))
        database.execute("CREATE TABLE documents(id TEXT PRIMARY KEY, dataset TEXT, publisher TEXT, digest TEXT, url TEXT, license_rank INT, words INT, record TEXT)")
        rejected = stack.enter_context(gzip.open(output / "rejections.jsonl.gz", "wt", encoding="utf-8"))
        def reject(dataset, path, index, row, reason, **extra):
            rejects[f"{dataset}:{reason}"] += 1
            rejected.write(json.dumps({"dataset": dataset, "input_file": str(path), "input_row": index,
                                       "original_id": row.get("id"), "url": row.get("url") or (row.get("metadata") or {}).get("url"),
                                       "reason": reason, **extra}, ensure_ascii=False) + "\n")
        for dataset, path, checksum in inputs:
            if nw.digest_file(path) != checksum:
                raise ValueError(f"Input checksum mismatch: {path}")
            before = path.stat()
            scanned = 0
            last_progress = time.monotonic()
            for index, row in iter_file(path, dataset):
                scanned += 1
                counts[f"scanned:{dataset}"] += 1
                if time.monotonic() - last_progress > 25:
                    database.commit()
                    print(f"Progress {path.name}: {scanned:,} rows; {counts[f'eligible_before_dedup:{dataset}']:,} eligible in source", flush=True)
                    last_progress = time.monotonic()
                if dataset == "hansard":
                    reason, date = d.hansard_decision(row, dt.date.fromisoformat(config["publication_cutoff"]).year)
                    if reason == "retain_candidate" and date > config["publication_cutoff"]:
                        reason = "publication_after_cutoff"
                    if reason != "retain_candidate":
                        reject(dataset, path, index, row, reason); continue
                    info = {"publication_date": date, "last_known_update_date": None,
                            "publication_date_evidence": "Date-bearing ID agrees with metadata.year", "url": None,
                            "canonical_url": "hansard:" + row["source"] + "/" + row["id"], "publisher": "UK Parliament",
                            "license": row["metadata"].get("license", "Open Parliament Licence"),
                            "whitespace_words": len(row["text"].split())}
                    if info["whitespace_words"] < config["minimum_words"]:
                        reject(dataset, path, index, row, "document_too_short_in_words"); continue
                else:
                    info, reason = preliminary(row, dataset, config)
                    if reason:
                        reject(dataset, path, index, row, reason); continue
                english = [detector.compute_language_confidence(sample, Language.ENGLISH) for sample in language_samples(row["text"])]
                if min(english) < config["english_confidence_minimum"]:
                    reject(dataset, path, index, row, "english_language_check_requires_review", english_sample_confidences=english); continue
                key = f"{dataset}/{path.name}/{index}"
                text_digest = d.text_hash(d.normalized(row["text"]))
                record = d.record(dataset, key.removeprefix(dataset + "/"), row["text"], "human_candidate", key,
                                  **info, original_id=row.get("id"), source=row.get("source", "cc-news"),
                                  metadata=row.get("metadata", {}), input_file=str(path), input_row=index, input_file_sha256=checksum,
                                  english_sample_confidences=english, collected_date=row.get("added"),
                                  license_status="unknown" if info["license"] == "unknown" else "declared_upstream_not_independently_verified",
                                  evidence="Selected source, publication/update rules and English/length checks; authorship inferred",
                                  version_evidence="Captured text revision is not independently dated; old publication date is insufficient",
                                  review_status="unreviewed", binary_target=None,
                                  flags=["human_origin_unconfirmed", "captured_version_date_unknown"] + (["license_unknown"] if info["license"] == "unknown" else []))
                database.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)", (key, dataset, info["publisher"], text_digest,
                                 info["canonical_url"], int(info["license"] == "unknown"), info["whitespace_words"], json.dumps(record, ensure_ascii=False)))
                groups.find(key)
                for identity in (("text", text_digest), ("url", info["canonical_url"])):
                    if identity in identities:
                        groups.union(key, identities[identity])
                    else:
                        identities[identity] = key
                counts[f"eligible_before_dedup:{dataset}"] += 1
                if scanned % 5000 == 0:
                    database.commit()
            database.commit()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Input changed during curation: {path}")
            manifests.append({"dataset": dataset, "path": str(path), "sha256": checksum, "rows": scanned})
            print(f"Curated {path.name}: scanned {scanned:,}; eligible {counts[f'eligible_before_dedup:{dataset}']:,} cumulative", flush=True)
        members, winners = collections.defaultdict(list), {}
        for key, rank, words in database.execute("SELECT id,license_rank,words FROM documents ORDER BY id"):
            group = groups.find(key)
            members[group].append(key)
            priority = (rank, -words, key)
            if group not in winners or priority < winners[group][0]:
                winners[group] = (priority, key)
        doc_handles = {dataset: stack.enter_context(gzip.open(output / f"{dataset}.jsonl.gz", "wt", encoding="utf-8"))
                       for dataset in ("common_pile_news", "cc_news", "hansard")}
        span_handles = {dataset: stack.enter_context(gzip.open(output / f"{dataset}.spans.jsonl.gz", "wt", encoding="utf-8")) for dataset in doc_handles}
        duplicate_handle = stack.enter_context(gzip.open(output / "duplicates.jsonl.gz", "wt", encoding="utf-8"))
        review = collections.defaultdict(list)
        for group in sorted(winners):
            key = winners[group][1]
            record = json.loads(database.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()[0])
            dataset = record["dataset"]
            bucket = int(d.text_hash(config["split_seed"] + ":" + group)[:16], 16) % 100
            split = "train" if bucket < 98 else "validation" if bucket == 98 else "test"
            record.update(split_group=group, split=split, equivalent_record_ids=sorted(members[group]))
            doc_handles[dataset].write(json.dumps(record, ensure_ascii=False) + "\n")
            for duplicate in members[group]:
                if duplicate != key:
                    duplicate_handle.write(json.dumps({"id": duplicate, "retained_id": key, "group": group,
                                                        "reason": "same_normalized_text_or_canonical_article_url"}) + "\n")
                    counts["duplicate_or_article_variant"] += 1
            spans = make_spans(record["text"], config["minimum_words"], config["maximum_span_words"])
            for start, end, words in spans:
                text = record["text"][start:end]
                span = {**record, "id": f"span/{key}/{start}-{end}", "parent_document_id": key,
                        "text": text, "text_sha256": d.text_hash(text), "whitespace_words": len(text.split()),
                        "character_range": [start, end]}
                span_handles[dataset].write(json.dumps(span, ensure_ascii=False) + "\n")
            stats = by_source[f"{dataset}:{record['publisher']}"]
            stats["documents"] += 1; stats["words"] += record["whitespace_words"]
            stats["spans"] += len(spans); stats["span_words"] += sum(x[2] for x in spans)
            stats[f"split:{split}"] += 1
            counts[f"retained_documents:{dataset}"] += 1; counts[f"retained_words:{dataset}"] += record["whitespace_words"]
            counts[f"retained_spans:{dataset}"] += len(spans); counts[f"retained_span_words:{dataset}"] += sum(x[2] for x in spans)
            sample = review[(dataset, record["publisher"])]
            sample.append((d.text_hash("review:" + key), key))
            sample.sort(); del sample[config["review_sample_per_publisher"]:]
        audit = stack.enter_context((output / "review_queue.jsonl").open("w", encoding="utf-8"))
        for stratum, selected in sorted(review.items()):
            for _, key in selected:
                record = json.loads(database.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()[0])
                group = groups.find(key)
                bucket = int(d.text_hash(config["split_seed"] + ":" + group)[:16], 16) % 100
                record.update(split_group=group, split="train" if bucket < 98 else "validation" if bucket == 98 else "test")
                audit.write(json.dumps({**record, "review_stratum": list(stratum), "reviewer": None,
                                        "confirmed_human_evidence": None, "confirmed_version_evidence": None,
                                        "confirmed_translation_workflow": None, "review_decision": None}, ensure_ascii=False) + "\n")
    manifest = {"processing_complete": True, "counts": dict(counts), "rejections": dict(rejects),
                "sources": {k: dict(v) for k, v in by_source.items()}, "input_files": manifests,
                "config_sha256": nw.digest_file(args.config), "script_sha256": nw.digest_file(Path(__file__)),
                "license_snapshot_sha256": nw.digest_file(output / "licenses.snapshot.json"),
                "dependency_versions": {p: importlib.metadata.version(p) for p in ("lingua-language-detector", "pyarrow")},
                "deduplication": "Transitive exact-normalized-text and canonical-URL groups; prefer declared-license copy then longest copy",
                "word_budget": None, "size_measure": "Whitespace-separated words; no model tokenizer",
                "limitations": ["Candidates are not certified human-only and are not detector negatives",
                                "Language confidence is not authorship confidence",
                                "Captured-version uncertainty remains after date filtering",
                                "No learned MT/LLM detector applied; near duplicates and embedded synthetic quotations remain",
                                "Deterministic publisher-stratified review queue is unreviewed; not a contamination estimate",
                                "Spans are exact contiguous slices; tails below minimum are omitted from span export but retained in full documents"]}
    manifest["outputs"] = {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    nw.save_json(output / "manifest.json", manifest)
    print(json.dumps({"counts": dict(counts), "rejections": dict(rejects)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/curation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/curated-v1")
    curate(parser.parse_args())


if __name__ == "__main__":
    main()
