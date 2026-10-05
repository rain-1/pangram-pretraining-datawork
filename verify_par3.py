"""Verify native paragraph provenance, process labels and work-level partitions."""

import argparse
import collections
import json
from pathlib import Path

import datawork as d
import newswork as nw
import par3work


def verify(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, checksum in manifest["outputs"].items():
        if Path(name).name != name or nw.digest_file(output / name) != checksum:
            raise ValueError(f"Release checksum mismatch: {name}")
    source = Path(manifest["input_file"])
    if nw.digest_file(source) != manifest["input_sha256"]:
        raise ValueError("Raw archive checksum mismatch")
    raw = par3work.load_data_only(source)
    records = d.read_jsonl(output / "records.jsonl")
    binary = d.read_jsonl(output / "binary_examples.jsonl")
    d.verify_records(records); d.verify_records(binary)
    config = manifest["config"]
    books, work_groups = {}, {}
    counts = collections.Counter()
    for row in records:
        book, index = raw[row["book_id"]], row["paragraph_index"]
        if row["source_text"] != book["source_paras"][index]:
            raise ValueError("Foreign source paragraph changed")
        if row["authorship"] == "machine_translation":
            original = book["gt_paras"][index]
            if row["engine"] != "Google Translate" or row["binary_target"] not in (1, None):
                raise ValueError("Machine process label disagrees")
        elif row["authorship"] == "human_translation":
            original = book["translator_data"][row["translator_id"]]["translator_paras"][index]
            if row["binary_target"] not in (0, None):
                raise ValueError("Human process label disagrees")
        else:
            raise ValueError("Unexpected process label")
        if row["text"] != original or row["text_sha256"] != d.text_hash(original):
            raise ValueError("English paragraph rewritten or joined")
        words = len(original.split())
        if not config["minimum_words"] <= words <= config["maximum_words"] or words != row["whitespace_words"]:
            raise ValueError("Word count/bounds mismatch")
        if row["source_file_sha256"] != manifest["input_sha256"] or row["intended_use"] != "translation_detector_research_only":
            raise ValueError("Source provenance/use mismatch")
        if books.setdefault(row["book_id"], row["split"]) != row["split"] or work_groups.setdefault(row["work_group"], row["split"]) != row["split"]:
            raise ValueError("Book or multi-volume work crosses partitions")
        counts[row["authorship"]] += 1
    if d.binary_examples(records) != binary or len(binary) != manifest["binary_examples"]:
        raise ValueError("Binary export deduplication/ambiguity handling mismatch")
    stage = output / "external_v1_manifest.json"
    if stage.exists():
        external = json.loads(stage.read_text())
        for key, name in [("scores_sha256", "external_v1_scores.jsonl"), ("evaluation_sha256", "external_v1_evaluation.json")]:
            if nw.digest_file(output / name) != external[key]:
                raise ValueError("External evaluation output mismatch")
        if external["examples_sha256"] != nw.digest_file(output / "binary_examples.jsonl") or external["model_retrained"]:
            raise ValueError("External evaluation scope mismatch")
    report = {"verified": True, "manifest_sha256": nw.digest_file(manifest_path), "verifier_sha256": nw.digest_file(Path(__file__)),
              "records": len(records), "binary_examples": len(binary), "labels": dict(counts), "book_entries_with_long_controls": len(books),
              "work_groups": len(work_groups), "checks": ["Raw and export hashes", "Exact intact native paragraphs", "Process labels", "Matched source provenance", "Word bounds", "Book/volume/exact-text split separation"],
              "scope": "Released process provenance and integrity; individual English translation rights and engine generation dates remain unverified"}
    nw.save_json(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    verify(parser.parse_args().output)
