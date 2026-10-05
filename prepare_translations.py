"""Prepare the exact long English Google Translate and human translation controls."""

import argparse
import collections
import json
from pathlib import Path
import shutil
import tempfile

import datawork as d
import par3work
import prepare as p

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "releases/google-translate-v1"
IDENTITY_KEYS = ("id", "text_sha256", "authorship", "binary_target", "engine", "book_id",
                 "paragraph_index", "translator_id", "source_language", "split", "split_group",
                 "work_group", "equivalent_record_ids")


def identities(rows):
    result = [{**{k: row[k] for k in IDENTITY_KEYS},
               "source_text_sha256": p.text_hash(row["source_text"])} for row in rows]
    return sorted(result, key=lambda row: row["id"])


def identity_hash(value):
    return p.text_hash(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def load_release(release=RELEASE):
    lock = json.loads((release / "release.json").read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        p.require(p.digest_file(release / p.plain_name(name)) == expected,
                  f"Translation release metadata changed: {name}")
    selection = json.loads((release / "selection.json").read_text(encoding="utf-8"))
    config = json.loads((release / "source.json").read_text(encoding="utf-8"))
    p.require(selection["schema_version"] == 1, "Unknown translation release schema")
    p.require(identity_hash(selection["identities"]) == selection["selection_sha256"],
              "Translation selection identity changed")
    return selection, config, lock


def check_examples(records, binary, selection, config):
    d.verify_records(records)
    d.verify_records(binary)
    p.require(d.binary_examples(records) == binary, "Binary deduplication differs")
    p.require(identities(binary) == selection["identities"], "Selected controls or partitions differ")
    p.require(len(binary) == selection["binary_examples"], "Translation count differs")
    books, works = {}, {}
    for row in records:
        p.require(row["intended_use"] == "translation_detector_research_only"
                  and row["language"] == "en", "Unexpected translation use/language")
        p.require(config["minimum_words"] <= len(row["text"].split()) <= config["maximum_words"]
                  and len(row["text"].split()) == row["whitespace_words"], "Translation word bounds differ")
        p.require(row["source_file_sha256"] == config["sha256"], "Translation source pin differs")
        if row["authorship"] == "machine_translation":
            p.require(row["engine"] == "Google Translate" and row["binary_target"] == 1,
                      "Unexpected machine label")
        else:
            p.require(row["authorship"] == "human_translation" and row["binary_target"] == 0,
                      "Unexpected human label")
        p.require(books.setdefault(row["book_id"], row["split"]) == row["split"], "Book crosses partitions")
        p.require(works.setdefault(row["work_group"], row["split"]) == row["split"], "Work crosses partitions")
    labels = {split: dict(collections.Counter(r["authorship"] for r in binary if r["split"] == split))
              for split in p.SPLITS}
    p.require(labels == selection["by_split_label"], "Translation split counts differ")
    p.require(dict(collections.Counter(r["source_language"] for r in binary))
              == selection["by_source_language"], "Translation source language counts differ")
    p.require({split: len({r["split_group"] for r in binary if r["split"] == split}) for split in p.SPLITS}
              == selection["independent_work_groups_per_split"], "Translation group counts differ")


def verify(output, selection, config, lock):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    p.require(manifest["release"] == selection["release"]
              and manifest["selection_sha256"] == selection["selection_sha256"]
              and manifest["selection_file_sha256"] == lock["files"]["selection.json"]
              and manifest["source_config_sha256"] == lock["files"]["source.json"]
              and manifest["input_sha256"] == config["sha256"], "Translation release identity differs")
    p.require(set(manifest["outputs"]) == {"records.jsonl", "binary_examples.jsonl", "licenses.snapshot.json"},
              "Unexpected translation export inventory")
    for name, expected in manifest["outputs"].items():
        p.require(p.digest_file(output / p.plain_name(name)) == expected, f"Translation export changed: {name}")
    p.require(manifest["outputs"]["licenses.snapshot.json"] == lock["files"]["licenses.snapshot.json"],
              "Translation rights inventory differs")
    records = d.read_jsonl(output / "records.jsonl")
    binary = d.read_jsonl(output / "binary_examples.jsonl")
    check_examples(records, binary, selection, config)
    p.require(manifest["binary_examples"] == len(binary), "Translation manifest count differs")
    return {"verified": True, "manifest_sha256": p.digest_file(manifest_path),
            "selection_sha256": selection["selection_sha256"], "binary_examples": len(binary),
            "labels": dict(collections.Counter(r["authorship"] for r in binary)),
            "scope": "Exact pinned process-labelled English paragraphs, source hashes and work partitions; individual rights and generator dates unverified"}


def prepare(output, cache, offline=False, verify_only=False, release=RELEASE):
    selection, config, lock = load_release(release)
    if output.exists() and any(output.iterdir()):
        report = verify(output, selection, config, lock)
        print(f"Translation controls already ready and verified: {output}", flush=True)
        return report
    p.require(not verify_only, f"No prepared translation controls at {output}")
    spec = {"cache_path": "par3.pkl", "kind": "file", "url": config["download_url"],
            "sha256": config["sha256"], "bytes": config["bytes"]}
    source = p.fetch_source(spec, cache, offline)
    print("Checking the archive data format and extracting intact paragraphs", flush=True)
    data = par3work.load_data_only(source)
    records, binary, counts = par3work.extract(data, config, config["sha256"])
    # Preserve the original acquisition location as relative provenance metadata.
    # The actual cache location is optional and is not part of selection identity.
    for row in records + binary:
        row["source_file"] = "data/raw/par3/par3.pkl"
    check_examples(records, binary, selection, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", suffix=".building", dir=output.parent))
    try:
        d.write_jsonl(temporary / "records.jsonl", records)
        d.write_jsonl(temporary / "binary_examples.jsonl", binary)
        shutil.copyfile(release / "licenses.snapshot.json", temporary / "licenses.snapshot.json")
        manifest = {"release": selection["release"], "processing_complete": True,
                    "selection_sha256": selection["selection_sha256"],
                    "selection_file_sha256": lock["files"]["selection.json"],
                    "source_config_sha256": lock["files"]["source.json"],
                    "original_manifest_sha256": selection["original_manifest_sha256"],
                    "input_sha256": config["sha256"], "binary_examples": len(binary), "counts": counts,
                    "by_split_label": selection["by_split_label"],
                    "intended_use": "translation_detector_research_only",
                    "outputs": {f.name: p.digest_file(f) for f in sorted(temporary.iterdir())}}
        (temporary / "manifest.json").write_bytes(p.json_bytes(manifest))
        report = verify(temporary, selection, config, lock)
        (temporary / "verification.json").write_bytes(p.json_bytes(report))
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Ready: {output}\n{len(binary):,} controls: "
          f"{report['labels']['machine_translation']:,} Google Translate, "
          f"{report['labels']['human_translation']:,} human translations", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ready/google-translate-v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/par3")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.output.resolve(), args.cache.resolve(), args.offline, args.verify_only)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Translation preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
