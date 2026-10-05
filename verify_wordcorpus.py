"""Independently verify assembled corpus integrity, counts, groups and license annotations."""

import argparse
import collections
import gzip
import json
from pathlib import Path

import datawork as d
import newswork as nw
import wordcorpus


def verify(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["processing_complete"]:
        raise ValueError("Assembly incomplete")
    for name, checksum in manifest["outputs"].items():
        if Path(name).name != name or nw.digest_file(output / name) != checksum:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    source = Path(manifest["input_curation"])
    if nw.digest_file(source / "manifest.json") != manifest["input_manifest_sha256"] or nw.digest_file(source / "verification.json") != manifest["input_verification_sha256"]:
        raise ValueError("Source curation/verification changed")
    policy = json.loads((source / "policy.json").read_text())
    licenses = json.loads((output / "licenses.snapshot.json").read_text())
    counts, by_source, annotations = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter()
    groups, texts, ids, lengths = {}, set(), set(), []
    for split in ("train", "validation", "test"):
        with gzip.open(output / f"{split}.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                words = len(row["text"].split())
                if row["split"] != split or row["dataset"] not in manifest["source_selection"]:
                    raise ValueError("Partition/source mismatch")
                if row["authorship"] != "human_candidate" or row["binary_target"] is not None:
                    raise ValueError("Unexpected authorship label")
                if words != row["whitespace_words"] or not policy["minimum_words"] <= words <= policy["maximum_span_words"] or row["text_sha256"] != d.text_hash(row["text"]):
                    raise ValueError("Word count, bounds or text hash mismatch")
                digest = d.text_hash(d.normalized(row["text"]))
                if row["id"] in ids or digest in texts:
                    raise ValueError("Duplicate ID or normalized passage text")
                ids.add(row["id"]); texts.add(digest)
                if groups.setdefault(row["split_group"], split) != split:
                    raise ValueError("Parent document/similarity group crosses partitions")
                checked = {"dataset": row["dataset"], "publisher": row.get("publisher")}
                wordcorpus.annotate_license(checked, licenses)
                if checked.get("publisher_license_observation") != row.get("publisher_license_observation"):
                    raise ValueError("Publisher observations disagree with the license snapshot")
                if row.get("license_review_status"):
                    annotations[f"{row['dataset']}:{row['publisher']}"] += 1
                counts["selected_words"] += words; counts["selected_spans"] += 1
                counts[f"{split}_words"] += words; counts[f"{split}_spans"] += 1
                by_source[row["dataset"]]["words"] += words; by_source[row["dataset"]]["spans"] += 1
                lengths.append(words)
    for key, value in counts.items():
        if manifest["counts"].get(key) != value:
            raise ValueError(f"Manifest count mismatch: {key}")
    if {k: dict(v) for k, v in by_source.items()} != manifest["by_source"]:
        raise ValueError("Source coverage mismatch")
    cap = manifest["word_budget"]
    if cap is not None and (counts["selected_words"] > cap or manifest["unfilled_words"] != cap - counts["selected_words"]):
        raise ValueError("Word budget violation")
    lengths.sort()
    report = {"verified": True, "manifest_sha256": nw.digest_file(manifest_path), "verifier_sha256": nw.digest_file(Path(__file__)),
              "counts": dict(counts), "parent_similarity_groups": len(groups), "license_annotations": dict(annotations),
              "word_lengths": {"min": min(lengths) if lengths else None, "median": lengths[len(lengths) // 2] if lengths else None, "max": max(lengths) if lengths else None},
              "scope": "Integrity, exact text uniqueness, groups and recorded licensing observations; no authorship or rights certification"}
    nw.save_json(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)
