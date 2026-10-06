"""Import released Twitter detector controls without rewriting their text."""

import collections
import csv
import hashlib
import json
from pathlib import Path

import datawork as d
import prepare as p

PRIORITY = {"train": 0, "validation": 1, "test": 2}


def row_base(spec, number, text, target, authorship, engine):
    p.require(isinstance(text, str), "Twitter text must be a string")
    return {
        "id": f"{spec['cache_path']}:{number}", "dataset": spec["dataset"],
        "text": text, "text_sha256": p.text_hash(text),
        "normalized_text_sha256": p.text_hash(d.normalized(text)),
        "whitespace_words": len(text.split()), "language": "en",
        "language_basis": "upstream English dataset; individual language unverified",
        "binary_target": target, "authorship": authorship, "engine": engine,
        "human_authorship_verified": False, "intended_use": "twitter_detector_research_only",
        "source_file": "data/raw/twitter/" + spec["cache_path"],
        "source_file_sha256": spec["sha256"], "source_revision": spec["revision"],
        "upstream_split": spec["split"],
    }


def read_tweepfake(path, spec):
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        p.require(set(reader.fieldnames or []) == {"screen_name", "text", "account.type", "class_type"},
                  "Unexpected TweepFake columns")
        for i, item in enumerate(reader):
            account, kind = item["account.type"], item["class_type"]
            p.require((account == "human" and kind == "human") or
                      (account == "bot" and kind in {"gpt2", "rnn", "others"}),
                      "Unexpected TweepFake class/account combination")
            p.require(bool(item["screen_name"]), "Missing TweepFake account")
            target = 0 if kind == "human" else (1 if kind in {"gpt2", "rnn"} else None)
            authorship = "human_reference_unverified" if target == 0 else (
                "machine_generation_account_label" if target == 1 else "bot_unknown_generation_method")
            row = row_base(spec, i, item["text"], target, authorship, kind if target == 1 else None)
            row.update(account=item["screen_name"], upstream_class=kind, upstream_account_type=account,
                       label_basis="Published account classification; individual posts not independently audited",
                       partition_unit="tweepfake-account:" + item["screen_name"],
                       exclusion_reason="unknown_bot_method" if target is None else None)
            rows.append(row)
    return rows


def read_unmasking(path, spec):
    with Path(path).open(encoding="utf-8") as handle:
        values = [json.loads(line) for line in handle if line.strip()]
    p.require(len(values) % 2 == 0, "Unmasking release must contain complete pairs")
    rows, humans = [], []
    for start in range(0, len(values), 2):
        human, generated = values[start:start + 2]
        p.require(set(human) == set(generated) == {"text", "label"}
                  and type(human["label"]) is int and human["label"] == 0
                  and type(generated["label"]) is int and generated["label"] == 1,
                  "Unmasking human/generated alternation changed")
        p.require(isinstance(human["text"], str), "Invalid Unmasking human reference")
        unit = "unmasking-pair:" + p.text_hash(d.normalized(human["text"]))
        humans.append(p.text_hash(human["text"]))
        for offset, item in enumerate((human, generated)):
            target = item["label"]
            row = row_base(spec, start + offset, item["text"], target,
                           "documented_llm_generation" if target else "human_reference_unverified",
                           spec["model"] if target else None)
            row.update(upstream_class=target, generation_variant=spec["model"], partition_unit=unit,
                       pair_index=start // 2, exclusion_reason=None,
                       label_basis="Author-released LLM generation" if target else
                       "TweetEval reference label; individual authorship not independently audited")
            rows.append(row)
    return rows, humans


def assign_groups(rows):
    """Keep accounts, paired prompts and normalized exact duplicates together."""
    uf, first = d.UnionFind(), {}
    for row in rows:
        unit = row["partition_unit"]
        uf.find(unit)
        key = row["normalized_text_sha256"]
        uf.union(unit, first.setdefault(key, unit))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[uf.find(row["partition_unit"])].append(row)

    # TweepFake's released random tweet splits share accounts. Reallocate whole
    # connected account groups, stratified by their eligible label composition.
    strata = collections.defaultdict(list)
    for group, members in groups.items():
        if any(r["dataset"] == "tweepfake" for r in members):
            targets = tuple(sorted({r["binary_target"] for r in members
                                    if r["dataset"] == "tweepfake" and r["binary_target"] is not None}))
            strata[targets].append(group)
    assignments = {}
    for group_ids in strata.values():
        ordered = sorted(group_ids, key=lambda g: p.text_hash("twitter-v1:" + g))
        held = max(1, round(len(ordered) * .15)) if len(ordered) >= 3 else 0
        for i, group in enumerate(ordered):
            assignments[group] = "test" if i < held else ("validation" if i < held * 2 else "train")
    for group, members in groups.items():
        candidates = [r["upstream_split"] for r in members if r["dataset"] == "unmasking"]
        if group in assignments:
            candidates.append(assignments[group])
        split = max(candidates, key=PRIORITY.__getitem__)
        for row in members:
            row.update(split=split, split_group=p.text_hash("twitter-v1-group:" + group))


def deduplicate(rows):
    by_text = collections.defaultdict(list)
    for row in rows:
        by_text[row["normalized_text_sha256"]].append(row)
    binary, quarantine = [], []
    for members in by_text.values():
        targets = {r["binary_target"] for r in members if r["binary_target"] is not None}
        if len(targets) > 1:
            for row in members:
                row["exclusion_reason"] = "conflicting_labels_for_identical_normalized_text"
            quarantine.extend(members)
            continue
        eligible = [r for r in members if r["binary_target"] is not None and d.normalized(r["text"])]
        if not eligible:
            for row in members:
                if not d.normalized(row["text"]):
                    row["exclusion_reason"] = "empty_text"
            quarantine.extend(members)
            continue
        chosen = min(eligible, key=lambda r: (r["dataset"], r["id"]))
        binary.append({**chosen, "equivalent_record_ids": sorted(r["id"] for r in members)})
        quarantine.extend(r for r in members if r["binary_target"] is None)
    return sorted(binary, key=lambda r: r["id"]), sorted(quarantine, key=lambda r: r["id"])


def extract(config, cache):
    rows, sequences = [], {}
    for spec in config["sources"]:
        if spec["dataset"] == "grokset" or spec.get("split") is None:
            continue
        path = cache / spec["cache_path"]
        p.check_source(path, spec)
        if spec["dataset"] == "tweepfake":
            rows.extend(read_tweepfake(path, spec))
        elif spec["dataset"] == "unmasking":
            part, humans = read_unmasking(path, spec)
            p.require(sequences.setdefault(spec["split"], humans) == humans,
                      "Human reference sequence differs between model variants")
            rows.extend(part)
        else:
            raise ValueError("Unknown Twitter dataset")
    rows.sort(key=lambda r: r["id"])
    assign_groups(rows)
    binary, quarantine = deduplicate(rows)
    validate(rows, binary, quarantine)
    return rows, binary, quarantine


def stable_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def summary(rows, binary, quarantine):
    return {"schema_version": 1, "release": "twitter-v1", "audit_records": len(rows),
            "binary_examples": len(binary), "quarantined_records": len(quarantine),
            "audit_sha256": stable_hash(rows), "selection_sha256": stable_hash(binary),
            "quarantine_sha256": stable_hash(quarantine),
            "by_dataset_label": {name: dict(collections.Counter(str(r["binary_target"]) for r in binary
                                                               if r["dataset"] == name))
                                 for name in ("tweepfake", "unmasking")},
            "by_split_label": {split: dict(collections.Counter(str(r["binary_target"]) for r in binary
                                                               if r["split"] == split)) for split in p.SPLITS},
            "groups_per_split": {split: len({r["split_group"] for r in binary if r["split"] == split})
                                 for split in p.SPLITS},
            "quarantine_reasons": dict(collections.Counter(r["exclusion_reason"] for r in quarantine))}


def validate(rows, binary, quarantine):
    ids, groups, units, texts = {}, {}, {}, {}
    for row in rows:
        p.require(row["id"] not in ids, "Duplicate Twitter record ID")
        ids[row["id"]] = row
        p.require(row["text_sha256"] == p.text_hash(row["text"])
                  and row["normalized_text_sha256"] == p.text_hash(d.normalized(row["text"]))
                  and row["whitespace_words"] == len(row["text"].split()), "Twitter text identity changed")
        p.require(row["split"] in p.SPLITS and row["human_authorship_verified"] is False
                  and row["intended_use"] == "twitter_detector_research_only", "Unexpected Twitter claim/use")
        for mapping, key in ((groups, row["split_group"]), (units, row["partition_unit"]),
                             (texts, row["normalized_text_sha256"])):
            p.require(mapping.setdefault(key, row["split"]) == row["split"], "Twitter leakage between splits")
    p.require(len({r["normalized_text_sha256"] for r in binary}) == len(binary), "Binary text duplicated")
    for row in binary:
        p.require(row["binary_target"] in (0, 1), "Unresolved reference entered binary data")
        p.require({k: v for k, v in row.items() if k != "equivalent_record_ids"} == ids[row["id"]],
                  "Binary reference differs from audit record")
        aliases = row["equivalent_record_ids"]
        p.require(bool(aliases) and all(i in ids and ids[i]["normalized_text_sha256"] == row["normalized_text_sha256"]
                  and ids[i]["binary_target"] in (None, row["binary_target"]) for i in aliases), "Invalid deduplication lineage")
    expected, excluded = deduplicate(rows)
    p.require(expected == binary and excluded == quarantine, "Twitter deduplication/quarantine differs")
