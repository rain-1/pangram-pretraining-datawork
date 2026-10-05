"""Assemble verified candidate spans into train/validation/test using word counts."""

import argparse
import collections
import contextlib
import gzip
import json
from pathlib import Path
import sqlite3

import datawork as d
import newswork as nw

SOURCES = ("common_pile_news", "cc_news", "hansard")
ROOT = Path(__file__).resolve().parent


def annotate_license(row, inventory):
    entry = next((x for x in inventory["datasets"] if x["id"] == row["dataset"]), {})
    observation = entry.get("publisher_observations", {}).get(row.get("publisher"))
    if observation:
        # Keep the original upstream field; distinguish newly observed evidence.
        row["publisher_license_observation"] = dict(observation)
        row["license_review_status"] = observation["status"]


def build(args):
    source = args.curated.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    license_inventory = json.loads((ROOT / "config/licenses.json").read_text())
    verified = json.loads((source / "verification.json").read_text())
    if not verified["verified"] or verified["manifest_sha256"] != nw.digest_file(source / "manifest.json"):
        raise ValueError("Input curation must have a matching verification report")
    if args.word_budget is not None and args.word_budget <= 0:
        raise ValueError("Word budget must be positive")
    if len(set(args.sources)) != len(args.sources) or not args.sources or not set(args.sources) <= set(SOURCES):
        raise ValueError("Choose distinct supported sources")
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be separate from curation inputs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    files = [(name, source / f"{name}.spans.jsonl.gz") for name in args.sources]
    for _, path in files:
        if nw.digest_file(path) != manifest["outputs"][path.name]:
            raise ValueError(f"Input span checksum mismatch: {path}")
    output.mkdir(parents=True, exist_ok=True)
    counts, rejected, coverage = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    with contextlib.ExitStack() as stack:
        database = stack.enter_context(contextlib.closing(sqlite3.connect(output / "selection.sqlite")))
        database.execute("CREATE TABLE candidates(id TEXT PRIMARY KEY, digest TEXT UNIQUE, priority TEXT, rank INT, words INT, record TEXT)")
        for dataset, path in files:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    annotate_license(row, license_inventory)
                    if row["dataset"] != dataset or row["authorship"] != "human_candidate" or row["binary_target"] is not None:
                        raise ValueError("Unexpected source or authorship")
                    words = len(row["text"].split())
                    if row["text_sha256"] != d.text_hash(row["text"]) or words != row["whitespace_words"]:
                        raise ValueError("Span text/word count mismatch")
                    counts[f"input_spans:{dataset}"] += 1
                    digest = d.text_hash(d.normalized(row["text"]))
                    rank = int(row["license"] == "unknown")
                    existing = database.execute("SELECT id,rank,record FROM candidates WHERE digest=?", (digest,)).fetchone()
                    if existing:
                        prior = json.loads(existing[2])
                        if prior["split"] != row["split"]:
                            raise ValueError("Exact span text crosses source partitions; regroup upstream before assembly")
                        rejected["duplicate_normalized_span"] += 1
                        if (rank, row["id"]) >= (existing[1], existing[0]):
                            continue
                        database.execute("DELETE FROM candidates WHERE id=?", (existing[0],))
                    priority = d.text_hash(args.seed + ":" + row["id"])
                    database.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?)", (row["id"], digest, priority, rank, words, json.dumps(row, ensure_ascii=False)))
            database.commit()
        handles = {split: stack.enter_context(gzip.open(output / f"{split}.jsonl.gz", "wt", encoding="utf-8")) for split in ("train", "validation", "test")}
        rejection_handle = stack.enter_context((output / "rejections.jsonl").open("w", encoding="utf-8"))
        for key, words, raw in database.execute("SELECT id,words,record FROM candidates ORDER BY priority,id"):
            if args.word_budget is not None and counts["selected_words"] + words > args.word_budget:
                rejected["exceeds_remaining_word_budget"] += 1
                rejection_handle.write(json.dumps({"id": key, "reason": "exceeds_remaining_word_budget", "words": words}) + "\n")
                continue
            row = json.loads(raw)
            split = row["split"]
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["selected_words"] += words
            counts["selected_spans"] += 1
            counts[f"{split}_words"] += words
            counts[f"{split}_spans"] += 1
            coverage[row["dataset"]]["words"] += words
            coverage[row["dataset"]]["spans"] += 1
        database.commit()
    nw.save_json(output / "licenses.snapshot.json", license_inventory)
    report = {
        "processing_complete": True, "input_curation": str(source),
        "input_manifest_sha256": nw.digest_file(source / "manifest.json"),
        "input_verification_sha256": nw.digest_file(source / "verification.json"),
        "input_spans": {str(p): nw.digest_file(p) for _, p in files},
        "script_sha256": nw.digest_file(Path(__file__)), "source_selection": args.sources,
        "selection_seed": args.seed, "size_measure": "Whitespace-separated words",
        "word_budget": args.word_budget,
        "unfilled_words": None if args.word_budget is None else args.word_budget - counts["selected_words"],
        "counts": dict(counts), "by_source": {k: dict(v) for k, v in coverage.items()}, "rejections": dict(rejected),
        "split_policy": "Preserve verified parent document partitions; exact span texts deduplicated globally",
        "limitations": ["Unreviewed human-origin candidates; no measured human-only guarantee",
                        "All source-specific license metadata and uncertainty preserved",
                        "No learned detector used for removal; captured-version and near-duplicate review remain",
                        "Cap applies across all splits; complete spans retained, never truncated or repeated",
                        "Full documents and span exports are alternatives; do not concatenate them as training inputs"]}
    report["outputs"] = {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    nw.save_json(output / "manifest.json", report)
    print(json.dumps({"counts": report["counts"], "by_source": report["by_source"], "rejections": report["rejections"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", type=Path, default=ROOT / "data/curated-v1")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--word-budget", type=int, help="Optional total word cap across all splits; default keeps all eligible spans")
    parser.add_argument("--seed", default="wordcorpus-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/corpus-words-v1")
    build(parser.parse_args())
