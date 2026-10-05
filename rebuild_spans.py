"""Rebuild only passage exports from immutable retained documents, recording lineage."""

import argparse
import collections
import gzip
import json
from pathlib import Path

import curate
import datawork as d
import newswork as nw


def rebuild(output):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    policy = json.loads((output / "policy.json").read_text())
    datasets = ("common_pile_news", "cc_news", "hansard")
    for dataset in datasets:
        path = output / f"{dataset}.jsonl.gz"
        if nw.digest_file(path) != manifest["outputs"][path.name]:
            raise ValueError("Retained document checksum mismatch")
    if nw.digest_file(output / "policy.json") != manifest["outputs"]["policy.json"]:
        raise ValueError("Policy checksum mismatch")
    counts, coverage = collections.Counter(), collections.defaultdict(collections.Counter)
    replacements = []
    for dataset in datasets:
        target = output / f"{dataset}.spans.jsonl.gz"
        temporary = target.with_suffix(target.suffix + ".part")
        with gzip.open(output / f"{dataset}.jsonl.gz", "rt", encoding="utf-8") as source, gzip.open(temporary, "wt", encoding="utf-8") as sink:
            for line in source:
                row = json.loads(line)
                ranges = curate.make_spans(row["text"], policy["minimum_words"], policy["maximum_span_words"])
                for start, end, words in ranges:
                    text = row["text"][start:end]
                    span = {**row, "id": f"span/{row['id']}/{start}-{end}", "parent_document_id": row["id"],
                            "text": text, "text_sha256": d.text_hash(text), "whitespace_words": words,
                            "character_range": [start, end]}
                    sink.write(json.dumps(span, ensure_ascii=False) + "\n")
                    counts[f"retained_spans:{dataset}"] += 1
                    counts[f"retained_span_words:{dataset}"] += words
                    coverage[f"{dataset}:{row['publisher']}"]["spans"] += 1
                    coverage[f"{dataset}:{row['publisher']}"]["span_words"] += words
        replacements.append((temporary, target))
        print(f"Rebuilt {dataset}: {counts[f'retained_spans:{dataset}']:,} spans", flush=True)
    original_hash = nw.digest_file(manifest_path)
    for temporary, target in replacements:
        temporary.replace(target)
    manifest["counts"].update(counts)
    for source, stats in coverage.items():
        manifest["sources"][source].update(stats)
    manifest["span_rebuild"] = {
        "prior_manifest_sha256": original_hash, "script_sha256": nw.digest_file(Path(__file__)),
        "span_function_script_sha256": nw.digest_file(Path(curate.__file__)),
        "reason": "Keep eligible final remainder together; preferred boundaries must meet minimum length",
        "full_documents_unchanged": True}
    manifest["outputs"] = {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in {"manifest.json", "verification.json"}}
    nw.save_json(manifest_path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    rebuild(parser.parse_args().output)
