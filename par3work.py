"""Acquire official Par3 and extract intact long Google/human translation controls."""

import argparse
import collections
import datetime as dt
import gzip
import json
import math
from pathlib import Path
import pickle
import pickletools
import re
import urllib.request

import datawork as d
import newswork as nw

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/par3.json"
RAW = ROOT / "data/raw/par3"
FORBIDDEN = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Executable pickle globals are forbidden")

    def persistent_load(self, pid):
        raise ValueError("External pickle references are forbidden")


def load_data_only(path):
    with path.open("rb") as handle:
        for opcode, _, position in pickletools.genops(handle):
            if opcode.name in FORBIDDEN:
                raise ValueError(f"Non-data pickle opcode {opcode.name} at {position}")
        if handle.read(1):
            raise ValueError("Unexpected trailing bytes in pickle release")
    with path.open("rb") as handle:
        return DataOnlyUnpickler(handle).load()


def fetch():
    config = json.loads(CONFIG.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "par3.pkl"
    if path.exists():
        if not config["sha256"] or nw.digest_file(path) != config["sha256"] or path.stat().st_size != config["bytes"]:
            raise ValueError("Existing archive checksum/size not established; not overwritten")
        print("Verified existing Par3 release", flush=True)
        return
    documents = {}
    for name in ("README.md", "LICENSE", "sample_par3.py"):
        url = f"https://raw.githubusercontent.com/{config['repo']}/{config['revision']}/{name}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pretraining-datawork/0.1"}), timeout=45) as response:
            body = response.read()
        saved = RAW / name
        saved.write_bytes(body)
        documents[name] = {"url": url, "sha256": nw.digest_file(saved)}
    with urllib.request.urlopen(urllib.request.Request(config["official_folder_url"], headers={"User-Agent": "Mozilla/5.0"}), timeout=45) as response:
        body = response.read()
    if config["file_id"].encode() not in body or b"par3.pkl" not in body:
        raise ValueError("Official download folder no longer contains expected release")
    (RAW / "official-folder.html").write_bytes(body)
    temporary = path.with_suffix(".pkl.part")
    with urllib.request.urlopen(urllib.request.Request(config["download_url"], headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("Download returned HTML rather than the archive")
        received = 0
        with temporary.open("wb") as handle:
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > config["bytes"]:
                    raise ValueError("Archive exceeds expected release size")
                handle.write(block)
                if received % (32 * 1024 * 1024) == 0:
                    print(f"Downloaded Par3 {received:,}/{config['bytes']:,} bytes", flush=True)
    if temporary.stat().st_size != config["bytes"]:
        raise ValueError("Archive size mismatch")
    observed = nw.digest_file(temporary)
    if config["sha256"] is not None and observed != config["sha256"]:
        raise ValueError("Pinned Par3 checksum mismatch")
    temporary.replace(path)
    config["sha256"] = observed
    nw.save_json(CONFIG, config)
    nw.save_json(RAW / "acquisition.json", {"downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "url": config["download_url"], "sha256": observed, "bytes": received, "hash_origin": config["checksum_policy"],
                 "documentation": documents, "folder_sha256": nw.digest_file(RAW / "official-folder.html")})
    print(f"Pinned official Par3 release: {observed}", flush=True)


def extract(data, config, archive_sha):
    if not isinstance(data, dict):
        raise ValueError("Expected a dictionary of books")
    records, counts = [], collections.Counter()
    for book, contents in sorted(data.items()):
        if not isinstance(book, str) or not isinstance(contents, dict):
            raise ValueError("Invalid book identity/schema")
        source, google, translators = contents["source_paras"], contents["gt_paras"], contents["translator_data"]
        if not isinstance(source, list) or not isinstance(google, list) or not isinstance(translators, dict) or len(source) != len(google):
            raise ValueError(f"Invalid paragraph alignment: {book}")
        if not translators or any(not isinstance(v["translator_paras"], list) or len(v["translator_paras"]) != len(source) for v in translators.values()):
            raise ValueError(f"Human alignment lengths disagree: {book}")
        counts["book_entries"] += 1
        work = next((group for pattern, group in config.get("work_group_patterns", {}).items() if re.fullmatch(pattern, book)), book)
        for index, (original, automatic) in enumerate(zip(source, google)):
            human = {name: value["translator_paras"][index] for name, value in translators.items()}
            counts["aligned_paragraphs"] += 1
            if not isinstance(original, str) or not isinstance(automatic, str) or not all(isinstance(x, str) or x is None for x in human.values()):
                raise ValueError(f"Non-string paragraph in {book}/{index}")
            if not original.strip() or not automatic.strip():
                counts["missing_source_or_google_paragraph"] += 1
                continue
            counts["missing_human_variants"] += sum(not x or not x.strip() for x in human.values())
            human = {k: v for k, v in human.items() if v and v.strip()}
            if not config["minimum_words"] <= len(automatic.split()) <= config["maximum_words"]:
                counts["google_paragraph_outside_word_bounds"] += 1
                continue
            eligible_human = {k: v for k, v in human.items() if config["minimum_words"] <= len(v.split()) <= config["maximum_words"]}
            if not eligible_human:
                counts["no_aligned_human_within_word_bounds"] += 1
                continue
            counts["retained_aligned_paragraphs"] += 1
            counts["human_variants_outside_word_bounds"] += len(human) - len(eligible_human)
            variants = {"google_translate": automatic, **eligible_human}
            for variant, text in sorted(variants.items()):
                machine = variant == "google_translate"
                records.append(d.record("par3", f"{book}/{index}/{variant}", text,
                              "machine_translation" if machine else "human_translation", work,
                              source_text=original, source_language=book.rsplit("_", 1)[-1],
                              engine="Google Translate" if machine else f"Par3 human {variant}",
                              book_id=book, work_group=work, paragraph_index=index, alignment_id=f"{book}/{index}", translator_id=None if machine else variant,
                              original_release_order="shuffled", source_file=str(RAW / "par3.pkl"), source_file_sha256=archive_sha,
                              domain="literary", example_granularity="intact_paragraph", generation_date=None,
                              evidence="Official Par3 README documents named Google output and human-written English translations",
                              evidence_url=f"https://github.com/{config['repo']}/blob/{config['revision']}/README.md",
                              license_status="individual_translation_rights_unverified; repository_MIT_is_software_license",
                              license="unknown_for_individual_translation_text", intended_use="translation_detector_research_only"))
    d.assign_splits(records)
    work_groups = sorted({r["split_group"] for r in records}, key=lambda group: d.text_hash(config.get("split_seed", "par3-work-groups-v2") + ":" + group))
    # Fixed group-count allocation avoids a tiny release accidentally getting
    # only one independent held-out work under per-group hash buckets.
    reserve = min(math.ceil(len(work_groups) * 0.1), (len(work_groups) - 1) // 2) if work_groups else 0
    test_groups, validation_groups = set(work_groups[:reserve]), set(work_groups[reserve:2 * reserve])
    for row in records:
        row["split"] = "test" if row["split_group"] in test_groups else "validation" if row["split_group"] in validation_groups else "train"
    d.verify_records(records)
    return records, d.binary_examples(records), dict(counts)


def build(output):
    config = json.loads(CONFIG.read_text())
    path = RAW / "par3.pkl"
    if not config["sha256"] or nw.digest_file(path) != config["sha256"]:
        raise ValueError("Par3 must match its pinned acquisition checksum")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    data = load_data_only(path)
    records, binary, counts = extract(data, config, config["sha256"])
    output.mkdir(parents=True, exist_ok=True)
    d.write_jsonl(output / "records.jsonl", records)
    d.write_jsonl(output / "binary_examples.jsonl", binary)
    nw.save_json(output / "licenses.snapshot.json", json.loads((ROOT / "config/licenses.json").read_text()))
    manifest = {"processing_complete": True, "input_file": str(path), "input_sha256": config["sha256"],
                "config": config, "config_sha256": nw.digest_file(CONFIG), "script_sha256": nw.digest_file(Path(__file__)),
                "datawork_sha256": nw.digest_file(ROOT / "datawork.py"), "pickle_policy": "All globals, callable construction and persistent references rejected",
                "counts": counts, "binary_examples": len(binary), "by_source_language": dict(collections.Counter(r["source_language"] for r in binary)),
                "by_split_label": {split: dict(collections.Counter(r["authorship"] for r in binary if r["split"] == split)) for split in ("train", "validation", "test")},
                "independent_books_per_split": {split: len({r["book_id"] for r in binary if r["split"] == split}) for split in ("train", "validation", "test")},
                "independent_work_groups_per_split": {split: len({r["split_group"] for r in binary if r["split"] == split}) for split in ("train", "validation", "test")},
                "limitations": ["Historical literary translations; not representative of modern original-English news",
                                "No release rows joined: shuffled order cannot establish adjacency",
                                "Only Google and human variants within word bounds and with a matched counterpart retained",
                                "Human translator identities and individual translation rights not independently established",
                                "Generation dates and exact Google engine version unknown",
                                "Identified multi-volume works/trilogies grouped together; partitions do not hold out source languages or authors"]}
    manifest["outputs"] = {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    nw.save_json(output / "manifest.json", manifest)
    print(json.dumps({"counts": counts, "binary_examples": len(binary), "by_split_label": manifest["by_split_label"]}, indent=2), flush=True)


def score(output, model_dir):
    import joblib
    prepared = json.loads((output / "manifest.json").read_text())
    if nw.digest_file(output / "binary_examples.jsonl") != prepared["outputs"]["binary_examples.jsonl"]:
        raise ValueError("Prepared examples checksum mismatch")
    model_manifest = json.loads((model_dir / "manifest.json").read_text())
    model_path = model_dir / "baseline.joblib"
    if nw.digest_file(model_path) != model_manifest["outputs"]["baseline.joblib"]:
        raise ValueError("Locally generated baseline checksum mismatch")
    local = joblib.load(model_path)
    records = d.read_jsonl(output / "binary_examples.jsonl")
    scores = []
    for start in range(0, len(records), 128):
        rows = records[start:start + 128]
        probabilities = local["classifier"].predict_proba(local["vectorizer"].transform([r["text"] for r in rows]))[:, 1]
        scores += [{"id": r["id"], "score": float(p)} for r, p in zip(rows, probabilities)]
    d.write_jsonl(output / "external_v1_scores.jsonl", scores)
    reports = {split: d.evaluate(records, scores, local["threshold"], split) for split in ("train", "validation", "test")}
    all_external = d.evaluate([{**r, "split": "test"} for r in records], scores, local["threshold"], "test")
    report = {"model_sha256": nw.digest_file(model_path), "examples_sha256": nw.digest_file(output / "binary_examples.jsonl"),
              "frozen_threshold": local["threshold"], "model_retrained": False, "scope": "External literary domain; no Par3 examples used to train or tune v1 baseline", "reports": reports,
              "all_external_examples": all_external, "document_cluster_bootstrap": cluster_metrics(records, scores, local["threshold"]),
              "scoring_script_sha256": nw.digest_file(Path(__file__)),
              "note": "Partitions belong to the Par3 release; even its train partition is external to v1; use its test partition for a consistent future comparison"}
    nw.save_json(output / "external_v1_evaluation.json", report)
    nw.save_json(output / "external_v1_manifest.json", {
        "examples_sha256": report["examples_sha256"], "model_sha256": report["model_sha256"],
        "script_sha256": report["scoring_script_sha256"], "model_retrained": False,
        "scores_sha256": nw.digest_file(output / "external_v1_scores.jsonl"),
        "evaluation_sha256": nw.digest_file(output / "external_v1_evaluation.json")})
    print(json.dumps({"all_external": all_external["metrics"].get("all"), "cluster_bootstrap": report["document_cluster_bootstrap"]}, indent=2), flush=True)


def cluster_metrics(records, scores, threshold, repetitions=2000, seed=42):
    """Resample whole connected work groups; aggregate counts before ratios."""
    import numpy as np
    prepared = {r["id"]: r["score"] for r in scores}
    counts = collections.defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for row in records:
        if row["id"] not in prepared:
            raise ValueError("Cluster evaluation requires complete scores")
        group = counts[row["split_group"]]
        predicted = prepared[row["id"]] >= threshold
        if row["binary_target"] == 0:
            group[0] += predicted; group[1] += 1
        elif row["binary_target"] == 1:
            group[2] += predicted; group[3] += 1
    if not counts:
        raise ValueError("No eligible work groups for cluster evaluation")
    matrix = np.asarray([counts[k] for k in sorted(counts)])
    random = np.random.default_rng(seed)
    resampled = matrix[random.integers(0, len(matrix), size=(repetitions, len(matrix)))].sum(axis=1)
    total = matrix.sum(axis=0)
    result = {"work_groups": len(matrix), "bootstrap_repetitions": repetitions, "seed": seed,
              "method": "Resample whole connected work groups with replacement; rates weighted by example counts within sampled groups",
              "scope": "Conditional on this literary benchmark and frozen model; not a corpus-wide FPR bound"}
    for name, numerator, denominator in [("human_fpr", 0, 1), ("machine_recall", 2, 3)]:
        valid = resampled[:, denominator] > 0
        values = resampled[valid, numerator] / resampled[valid, denominator]
        result[name] = {"value": float(total[numerator] / total[denominator]) if total[denominator] else None,
                        "cluster_percentile95": np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None,
                        "valid_resamples": int(valid.sum())}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "build", "score"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/detector-par3-v1")
    parser.add_argument("--model", type=Path, default=ROOT / "data/detector-v1")
    args = parser.parse_args()
    if args.command == "fetch": fetch()
    elif args.command == "build": build(args.output)
    else: score(args.output, args.model)
