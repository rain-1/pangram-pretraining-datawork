"""Download pinned TweepFake and Unmasking Twitter detector controls."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import prepare as p
import twitterwork as tw

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "releases/twitter-v1"


def load_release(release=RELEASE):
    lock = json.loads((release / "release.json").read_text(encoding="utf-8"))
    p.require(set(lock["files"]) == {"source.json", "selection.json", "licenses.snapshot.json"},
              "Unexpected Twitter release inventory")
    for name, expected in lock["files"].items():
        p.require(p.digest_file(release / p.plain_name(name)) == expected, f"Twitter release changed: {name}")
    config = json.loads((release / "source.json").read_text(encoding="utf-8"))
    selection = json.loads((release / "selection.json").read_text(encoding="utf-8"))
    p.require(config["schema_version"] == selection["schema_version"] == 1
              and config["release"] == selection["release"] == "twitter-v1", "Unknown Twitter release")
    return selection, config, lock


def verify(output, selection, config, lock):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    names = {"records.jsonl.gz", "quarantine.jsonl.gz", "licenses.snapshot.json"}
    names.update(split + ".jsonl.gz" for split in p.SPLITS)
    p.require(set(manifest["outputs"]) == names and manifest["release"] == "twitter-v1"
              and manifest["release_files"] == lock["files"], "Twitter export release differs")
    for name, digest in manifest["outputs"].items():
        p.require(p.digest_file(output / p.plain_name(name)) == digest, f"Twitter export changed: {name}")
    p.require(manifest["outputs"]["licenses.snapshot.json"] == lock["files"]["licenses.snapshot.json"],
              "Twitter rights inventory changed")
    rows = p.read_rows(output / "records.jsonl.gz")
    binary = []
    for split in p.SPLITS:
        part = p.read_rows(output / (split + ".jsonl.gz"))
        p.require(all(r["split"] == split for r in part), "Twitter row in wrong partition file")
        binary.extend(part)
    binary.sort(key=lambda r: r["id"])
    quarantine = p.read_rows(output / "quarantine.jsonl.gz")
    tw.validate(rows, binary, quarantine)
    p.require(tw.summary(rows, binary, quarantine) == selection == manifest["summary"],
              "Twitter selection or partitions changed")
    return {"verified": True, "selection_sha256": selection["selection_sha256"],
            "manifest_sha256": p.digest_file(output / "manifest.json"),
            "binary_examples": len(binary), "by_dataset_label": selection["by_dataset_label"],
            "scope": "Pinned text, process labels, deduplication and group splits; individual human authorship and tweet rights unverified"}


def prepare(output, cache, offline=False, verify_only=False, release=RELEASE):
    selection, config, lock = load_release(release)
    if output.exists() and any(output.iterdir()):
        report = verify(output, selection, config, lock)
        print(f"Twitter controls already verified: {output}", flush=True)
        return report
    p.require(not verify_only, f"No prepared Twitter controls at {output}")
    for spec in config["sources"]:
        # Grok's text is not in its release. Its IDs are a separate optional job.
        if spec["cache_path"] != "grokset-dehydrated.json":
            p.fetch_source(spec, cache, offline)
    rows, binary, quarantine = tw.extract(config, cache)
    p.require(tw.summary(rows, binary, quarantine) == selection, "Twitter extraction differs from frozen release")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", suffix=".building", dir=output.parent))
    try:
        p.write_rows(temporary / "records.jsonl.gz", rows)
        p.write_rows(temporary / "quarantine.jsonl.gz", quarantine)
        for split in p.SPLITS:
            p.write_rows(temporary / (split + ".jsonl.gz"), [r for r in binary if r["split"] == split])
        shutil.copyfile(release / "licenses.snapshot.json", temporary / "licenses.snapshot.json")
        manifest = {"release": "twitter-v1", "release_files": lock["files"], "summary": selection,
                    "intended_use": "twitter_detector_research_only",
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
    print(f"Ready: {output}\n{len(binary):,} deduplicated Twitter detector controls", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ready/twitter-v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/twitter")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.output.resolve(), args.cache.resolve(), args.offline, args.verify_only)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Twitter preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
