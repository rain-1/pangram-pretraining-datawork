"""Verify curation artifacts, exact source slices, provenance and split separation."""

import argparse
import collections
import gzip
import json
from pathlib import Path
import sqlite3

import datawork as d
import newswork as nw


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def verify(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["processing_complete"]:
        raise ValueError("Curation did not complete")
    for name, expected in manifest["outputs"].items():
        if Path(name).name != name or nw.digest_file(output / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    policy = json.loads((output / "policy.json").read_text())
    doc_index, span_index = {}, {}
    counts = collections.Counter()
    duplicates = collections.Counter()
    all_ids = set()
    connection = sqlite3.connect(f"file:{output.resolve() / 'inventory.sqlite'}?mode=ro", uri=True)
    try:
        for dataset in ("common_pile_news", "cc_news", "hansard"):
            previous_ends = {}
            for row in rows(output / f"{dataset}.jsonl.gz"):
                key = row["id"]
                if key in all_ids:
                    raise ValueError("Duplicate document ID")
                all_ids.add(key)
                if row["dataset"] != dataset or row["authorship"] != "human_candidate" or row["binary_target"] is not None:
                    raise ValueError("Unexpected source/authorship label")
                if row["text_sha256"] != d.text_hash(row["text"]) or row["whitespace_words"] != len(row["text"].split()):
                    raise ValueError(f"Text hash/word count mismatch: {key}")
                if row["publication_date"] > policy["publication_cutoff"] or (row["last_known_update_date"] or "") > policy["publication_cutoff"]:
                    raise ValueError("Date policy violation")
                if row["whitespace_words"] < policy["minimum_words"]:
                    raise ValueError("Short document")
                stored = json.loads(connection.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()[0])
                if stored["text"] != row["text"] or stored["input_file_sha256"] != row["input_file_sha256"] or stored["metadata"] != row["metadata"]:
                    raise ValueError("Original text/provenance changed")
                for identity in (("group", row["split_group"]), ("text", d.text_hash(d.normalized(row["text"]))), ("url", row["canonical_url"])):
                    if identity in doc_index:
                        raise ValueError(f"Duplicate document/group/article retained: {key}")
                    doc_index[identity] = row["split"]
                doc_index[("id", key)] = {k: row[k] for k in ("split_group", "split", "dataset", "license", "metadata")}
                counts[f"retained_documents:{dataset}"] += 1
                counts[f"retained_words:{dataset}"] += row["whitespace_words"]
            for row in rows(output / f"{dataset}.spans.jsonl.gz"):
                key, parent = row["id"], row["parent_document_id"]
                if key in all_ids:
                    raise ValueError("Duplicate span ID")
                all_ids.add(key)
                info = doc_index.get(("id", parent))
                if info is None or any(row[k] != v for k, v in info.items()):
                    raise ValueError("Span partition/provenance disagrees with parent")
                stored = json.loads(connection.execute("SELECT record FROM documents WHERE id=?", (parent,)).fetchone()[0])
                start, end = row["character_range"]
                if not 0 <= start < end <= len(stored["text"]) or stored["text"][start:end] != row["text"]:
                    raise ValueError(f"Span is not an exact source slice: {key}")
                if start != previous_ends.get(parent, 0):
                    raise ValueError("Noncontiguous span coverage")
                previous_ends[parent] = end
                words = len(row["text"].split())
                if row["text_sha256"] != d.text_hash(row["text"]) or row["whitespace_words"] != words:
                    raise ValueError("Span hash/word count mismatch")
                if not policy["minimum_words"] <= words <= policy["maximum_span_words"]:
                    raise ValueError("Span word bounds violation")
                digest = d.text_hash(d.normalized(row["text"]))
                if digest in span_index:
                    duplicates["same_split" if span_index[digest] == row["split"] else "across_splits"] += 1
                span_index[digest] = row["split"]
                counts[f"retained_spans:{dataset}"] += 1
                counts[f"retained_span_words:{dataset}"] += words
            for parent, end in previous_ends.items():
                stored = json.loads(connection.execute("SELECT record FROM documents WHERE id=?", (parent,)).fetchone()[0])
                if len(stored["text"][end:].split()) >= policy["minimum_words"]:
                    raise ValueError("Eligible trailing passage omitted")
        for key, value in counts.items():
            if manifest["counts"].get(key) != value:
                raise ValueError(f"Manifest count mismatch: {key}")
        for dataset in ("common_pile_news", "cc_news", "hansard"):
            rejected = sum(v for k, v in manifest["rejections"].items() if k.startswith(dataset + ":"))
            if rejected + manifest["counts"][f"eligible_before_dedup:{dataset}"] != manifest["counts"][f"scanned:{dataset}"]:
                raise ValueError("Scanned/rejected/eligible counts do not reconcile")
        if duplicates["across_splits"]:
            raise ValueError(f"Exact span text crosses splits: {duplicates['across_splits']}")
    finally:
        connection.close()
    result = {"verified": True, "counts": dict(counts), "duplicate_span_texts": dict(duplicates),
              "manifest_sha256": nw.digest_file(manifest_path), "verifier_sha256": nw.digest_file(Path(__file__)),
              "scope": "Artifact integrity and curation rules only; not human-authorship certification or rights review"}
    nw.save_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)
