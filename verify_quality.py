"""Verify quality-v2 slices, decisions, counts and detected similarity partitions."""

import argparse
import collections
import contextlib
import functools
import json
from pathlib import Path
import sqlite3

import datawork as d
import newswork as nw
import quality


def verify(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["processing_complete"]:
        raise ValueError("Quality processing incomplete")
    audit = Path(manifest["audit_directory"])
    for directory, recorded in ((output, manifest["outputs"]), (audit, manifest["audit_outputs"])):
        for name, checksum in recorded.items():
            if Path(name).name != name or nw.digest_file(directory / name) != checksum:
                raise ValueError(f"Artifact checksum mismatch: {directory / name}")
    source = Path(manifest["input_curation"])
    if nw.digest_file(source / "manifest.json") != manifest["input_manifest_sha256"]:
        raise ValueError("Source curation manifest changed")
    policy = json.loads((output / "policy.json").read_text())
    counts, documents, full_digests, span_digests, ids, groups = collections.Counter(), {}, set(), {}, set(), {}
    with contextlib.closing(sqlite3.connect(f"file:{source / 'inventory.sqlite'}?mode=ro", uri=True)) as originals, contextlib.closing(sqlite3.connect(f"file:{audit / 'documents.sqlite'}?mode=ro", uri=True)) as inventory:
        @functools.lru_cache(maxsize=64)
        def staging(key):
            result = inventory.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()
            if result is None:
                raise ValueError(f"Missing staged record {key}")
            return json.loads(result[0])

        for dataset in quality.DATASETS:
            previous_ends = {}
            for row in quality.read_rows(output / f"{dataset}.jsonl.gz"):
                key = row["id"]
                if key in ids or row["dataset"] != dataset:
                    raise ValueError("Duplicate ID or source mismatch")
                ids.add(key)
                if row["authorship"] != "human_candidate" or row["binary_target"] is not None:
                    raise ValueError("Candidate provenance promoted to a classifier label")
                original = json.loads(originals.execute("SELECT record FROM documents WHERE id=?", (key,)).fetchone()[0])
                lineage = row["quality_v2"]
                start, end = lineage["source_character_range"]
                if not 0 <= start < end <= len(original["text"]) or row["text"] != original["text"][start:end]:
                    raise ValueError("Cleaned body is not an exact original slice")
                if lineage["source_text_sha256"] != original["text_sha256"] or row["text_sha256"] != d.text_hash(row["text"]):
                    raise ValueError("Original/retained text hash mismatch")
                if row["metadata"] != original["metadata"] or row["license"] != original["license"]:
                    raise ValueError("Original metadata/license lost")
                if staging(key)["text"] != row["text"] or staging(key)["split"] != lineage["source_split"]:
                    raise ValueError("Staged source text/partition changed")
                words = len(row["text"].split())
                if words != row["whitespace_words"] or words < policy["minimum_words"]:
                    raise ValueError("Document word count/bounds mismatch")
                digest = d.text_hash(d.normalized(row["text"]))
                if digest in full_digests:
                    raise ValueError("Duplicate full document text")
                full_digests.add(digest)
                if groups.setdefault(row["split_group"], row["split"]) != row["split"]:
                    raise ValueError("Parent similarity group crosses splits")
                documents[key] = {k: row[k] for k in ("split", "split_group", "dataset", "metadata", "license")}
                counts[f"retained_documents:{dataset}"] += 1
                counts[f"retained_words:{dataset}"] += words
            for row in quality.read_rows(output / f"{dataset}.spans.jsonl.gz"):
                key, parent = row["id"], row["parent_document_id"]
                if key in ids or parent not in documents:
                    raise ValueError("Duplicate/orphan span ID")
                ids.add(key)
                if any(row[k] != v for k, v in documents[parent].items()):
                    raise ValueError("Span partition/license/provenance disagrees with parent")
                body = staging(parent)["text"]
                start, end = row["character_range"]
                if not 0 <= start < end <= len(body) or body[start:end] != row["text"] or start != previous_ends.get(parent, 0):
                    raise ValueError("Span is not an exact contiguous parent slice")
                if row["original_character_range"] != [start, end]:
                    raise ValueError("Original offsets disagree with preserved prefix")
                previous_ends[parent] = end
                words = len(row["text"].split())
                if words != row["whitespace_words"] or not policy["minimum_words"] <= words <= policy["maximum_span_words"] or row["text_sha256"] != d.text_hash(row["text"]):
                    raise ValueError("Span length/hash mismatch")
                digest = d.text_hash(d.normalized(row["text"]))
                if digest in span_digests and span_digests[digest] != row["split"]:
                    raise ValueError("Exact passage text crosses partitions")
                span_digests[digest] = row["split"]
                counts[f"retained_spans:{dataset}"] += 1
                counts[f"retained_span_words:{dataset}"] += words
            for parent, end in previous_ends.items():
                if len(staging(parent)["text"][end:].split()) >= policy["minimum_words"]:
                    raise ValueError("Eligible final passage omitted")
        checked_pairs = 0
        for pair in quality.read_rows(audit / "similarity_pairs.jsonl.gz"):
            first, second = pair["first"], pair["second"]
            a, b = staging(first), staging(second)
            score = quality.jaccard(quality.shingles(a["text"], policy["shingle_words"]), quality.shingles(b["text"], policy["shingle_words"]))
            if abs(score - pair["shingle_jaccard"]) > 1e-12 or score < policy["group_jaccard_minimum"]:
                raise ValueError("Similarity evidence does not match actual word shingles")
            if first in documents and second in documents and documents[first]["split"] != documents[second]["split"]:
                raise ValueError("Detected near copies cross splits")
            checked_pairs += 1
        decisions = d.read_jsonl(audit / "near_duplicate_decisions.jsonl")
        for decision in decisions:
            removed, retained = decision["removed_id"], decision["retained_id"]
            if removed in documents or retained not in documents:
                raise ValueError("Invalid near-duplicate removal target")
            a, b = staging(removed), staging(retained)
            score = quality.jaccard(quality.shingles(a["text"], policy["shingle_words"]), quality.shingles(b["text"], policy["shingle_words"]))
            ratio = min(a["whitespace_words"], b["whitespace_words"]) / max(a["whitespace_words"], b["whitespace_words"])
            if score < policy["dedup_jaccard_minimum"] or ratio < policy["dedup_word_ratio_minimum"] or abs(score - decision["shingle_jaccard"]) > 1e-12:
                raise ValueError("Deletion lacks a direct confirmed near-duplicate retained copy")
    for key, expected in manifest["counts"].items():
        if key.startswith(("retained_documents:", "retained_words:", "retained_spans:", "retained_span_words:")) and counts[key] != expected:
            raise ValueError(f"Count mismatch: {key}")
    if manifest["counts"]["input_documents"] != sum(manifest["rejections"].values()) + len(documents) + len(decisions):
        raise ValueError("Input/retained/rejection/deletion counts do not reconcile")
    report = {"verified": True, "manifest_sha256": nw.digest_file(manifest_path), "verifier_sha256": nw.digest_file(Path(__file__)),
              "counts": dict(counts), "confirmed_pairs_recomputed": checked_pairs, "direct_deletions_recomputed": len(decisions),
              "scope": "Integrity, original slices, detected near copies and split rules; not human-authorship or rights certification"}
    nw.save_json(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)
