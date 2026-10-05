"""Build and verify a conservative human-origin core from unchanged v2 passages."""

import argparse
import collections
import contextlib
import datetime as dt
import gzip
import json
from pathlib import Path
import re

import datawork as d
import newswork as nw

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/human_core.json"
SPLITS = ("train", "validation", "test")
ATTRIBUTION = re.compile(r"(?m)^[ \t]*((?:Lord|Baroness|Earl|Viscount|The\s+(?:Lord|Earl|Viscount|Baroness|Archbishop|Bishop))[^:\n]{1,110}):[ \t]*")
NOTICE = re.compile(r"\b(?:computer[ -]generated|machine[ -]generated|automatically\s+generated|machine\s+translation|automatically\s+translated|google\s+translate|deepl|babelfish|systran|chatgpt|gpt[ -]?[234])\b", re.I)
PARSER_NOTE = re.compile(r"\b(?:question\s+number\s+missing\s+in\s+hansard|possibly\s+truncated\s+question|parlparse\s+(?:error|warning)|no\s+(?:question|speaker|answer)\s+(?:found|available))\b", re.I)
LICENSE_URL = "https://www.parliament.uk/site-information/copyright-parliament/open-parliament-licence/"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def source_gate(row, policy):
    if (row.get("dataset"), row.get("source"), row.get("publisher")) != (policy["dataset"], policy["source"], policy["publisher"]):
        return "source_outside_strict_core"
    if row.get("language") != "en" or row.get("metadata", {}).get("language") != "en":
        return "missing_english_provenance"
    try:
        date = dt.date.fromisoformat(row["publication_date"])
    except (ValueError, TypeError, KeyError):
        return "missing_or_invalid_publication_date"
    identity = re.fullmatch(r"lordswrans(\d{4}-\d{2}-\d{2})[a-z]*", str(row.get("original_id", "")))
    if not identity or identity.group(1) != date.isoformat() or str(row.get("metadata", {}).get("year")) != str(date.year):
        return "date_id_year_disagreement"
    if date > dt.date.fromisoformat(policy["publication_cutoff"]):
        return "publication_after_strict_cutoff"
    if row.get("last_known_update_date") and row["last_known_update_date"] > policy["publication_cutoff"]:
        return "known_revision_after_strict_cutoff"
    if row.get("text_version_date") and row["text_version_date"] > policy["publication_cutoff"]:
        return "known_text_version_after_strict_cutoff"
    if LICENSE_URL not in row.get("license", "") or row.get("license_review_status"):
        return "missing_or_disputed_parliamentary_license"
    if row.get("authorship") != "human_candidate" or row.get("binary_target") is not None or row.get("engine"):
        return "unexpected_authorship_or_engine"
    return None


def prose_metrics(text):
    words = text.split()
    lines = [line for line in text.splitlines() if line.strip()]
    def numeric_line(line):
        tokens = line.split()
        return sum(any(c.isdigit() for c in t) for t in tokens) / len(tokens) >= 0.4
    return {
        "digit_token_fraction": sum(any(c.isdigit() for c in t) for t in words) / max(1, len(words)),
        "numeric_line_fraction": sum(numeric_line(line) for line in lines) / max(1, len(lines)),
        "tabular_line_fraction": sum(bool(re.match(r"^\s*\t\s*\t", line)) for line in lines) / max(1, len(lines)),
        "alphabetic_token_fraction": sum(any(c.isalpha() for c in t) for t in words) / max(1, len(words)),
    }


def decision(row, parents, policy):
    reason = source_gate(row, policy)
    if reason:
        return reason, None
    parent = parents.get(row.get("parent_document_id"))
    if parent is None:
        raise ValueError("Missing source parent")
    if source_gate(parent, policy):
        raise ValueError("Passage and parent source policy disagree")
    start, end = row["character_range"]
    if parent["text"][start:end] != row["text"] or parent["split_group"] != row["split_group"] or parent["split"] != row["split"]:
        raise ValueError("Passage body or partition differs from parent")
    if NOTICE.search(parent["text"]):
        return "parent_generation_or_translation_mention_requires_review", None
    if PARSER_NOTE.search(parent["text"]):
        return "parent_extraction_annotation_requires_review", None
    text = row["text"]
    words = len(text.split())
    if not policy["minimum_words"] <= words <= policy["maximum_words"]:
        return "outside_word_bounds", None
    if row["text_sha256"] != d.text_hash(text) or words != row["whitespace_words"]:
        raise ValueError("Passage integrity differs")
    attributions = list(ATTRIBUTION.finditer(text))
    names = {m.group(1).strip() for m in attributions}
    if len(names) < policy["minimum_distinct_named_participants"]:
        return "insufficient_named_participants", None
    questions = [m for m in attributions if re.match(r"asked\b|To\s+ask\b", text[m.end():], re.I)]
    if not any(any(a.start() > q.start() and a not in questions for a in attributions) for q in questions):
        return "no_explicit_named_question_and_answer", None
    metrics = prose_metrics(text)
    for key in ("digit_token_fraction", "numeric_line_fraction", "tabular_line_fraction"):
        if metrics[key] > policy["maximum_" + key]:
            return "prose_screen:" + key, metrics
    if metrics["alphabetic_token_fraction"] < policy["minimum_alphabetic_token_fraction"]:
        return "prose_screen:alphabetic_token_fraction", metrics
    return None, {"basis": "high_likelihood_from_historical_named_institutional_provenance",
                  "rationale": policy["rationale"], "publication_cutoff": policy["publication_cutoff"],
                  "named_participants": sorted(names), "prose_metrics": metrics,
                  "evidence_urls": policy["evidence_urls"],
                  "individual_authorship_verified": False, "captured_text_version_verified": False,
                  "learned_detector_used": False}


def checked_inputs(source, parent_dir):
    manifest = read(source / "manifest.json")
    verification = read(source / "verification.json")
    if not verification["verified"] or verification["manifest_sha256"] != nw.digest_file(source / "manifest.json"):
        raise ValueError("Source corpus verification missing or stale")
    parent_manifest = read(parent_dir / "manifest.json")
    parent_verification = read(parent_dir / "verification.json")
    if (manifest["input_manifest_sha256"] != nw.digest_file(parent_dir / "manifest.json")
            or not parent_verification["verified"] or parent_verification["manifest_sha256"] != nw.digest_file(parent_dir / "manifest.json")):
        raise ValueError("Parent curation verification missing or stale")
    hashes = {}
    for path, expected in [(source / (s + ".jsonl.gz"), manifest["outputs"][s + ".jsonl.gz"]) for s in SPLITS] + [(parent_dir / "hansard.jsonl.gz", parent_manifest["outputs"]["hansard.jsonl.gz"])]:
        if nw.digest_file(path) != expected:
            raise ValueError(f"Input checksum mismatch: {path}")
        hashes[str(path)] = expected
    parents = {r["id"]: r for r in rows(parent_dir / "hansard.jsonl.gz")}
    return parents, hashes


def build(args):
    source, parent_dir, output = args.input.resolve(), args.parents.resolve(), args.output.resolve()
    if output in (source, parent_dir) or source in output.parents or parent_dir in output.parents:
        raise ValueError("Use a separate output directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    parents, hashes = checked_inputs(source, parent_dir)
    policy = read(args.config)
    evidence = ROOT / "data/raw/human-core-evidence"
    evidence_hashes = {str(p.relative_to(ROOT)): nw.digest_file(p) for p in sorted(evidence.iterdir()) if p.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    counts, excluded, years = collections.Counter(), collections.Counter(), collections.Counter()
    groups = set()
    with contextlib.ExitStack() as stack:
        handles = {s: stack.enter_context(gzip.open(output / (s + ".jsonl.gz"), "wt", encoding="utf-8")) for s in SPLITS}
        rejected = stack.enter_context(gzip.open(output / "exclusions.jsonl.gz", "wt", encoding="utf-8"))
        for split in SPLITS:
            for row in rows(source / (split + ".jsonl.gz")):
                counts["input_spans"] += 1
                reason, evidence_record = decision(row, parents, policy)
                if reason:
                    excluded[reason] += 1
                    rejected.write(json.dumps({"id": row["id"], "dataset": row["dataset"], "reason": reason}) + "\n")
                    continue
                row["human_core_selection"] = evidence_record
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["selected_spans"] += 1
                counts["selected_words"] += row["whitespace_words"]
                counts[split + "_spans"] += 1
                counts[split + "_words"] += row["whitespace_words"]
                years[row["publication_date"][:4]] += 1
                groups.add(row["parent_document_id"])
    nw.save_json(output / "policy.json", policy)
    nw.save_json(output / "licenses.snapshot.json", read(ROOT / "config/licenses.json"))
    (output / "ATTRIBUTION.txt").write_text("Contains Parliamentary information licensed under the Open Parliament Licence v3.0.\n" + LICENSE_URL + "\n\nDerived from Common Pile UK Hansard via ParlParse.\nhttps://huggingface.co/datasets/common-pile/uk_hansard\n")
    manifest = {"processing_complete": True, "input_corpus": str(source), "parent_curation": str(parent_dir),
                "input_manifest_sha256": nw.digest_file(source / "manifest.json"), "input_verification_sha256": nw.digest_file(source / "verification.json"),
                "parent_manifest_sha256": nw.digest_file(parent_dir / "manifest.json"), "input_files": hashes,
                "policy_sha256": nw.digest_file(args.config), "script_sha256": nw.digest_file(Path(__file__)),
                "evidence_files": evidence_hashes, "counts": dict(counts), "parent_documents": len(groups),
                "exclusions": dict(excluded), "passages_by_year": dict(sorted(years.items())),
                "split_policy": "Unchanged v2 parent/similarity group partitions; do not add this overlapping subset to v2",
                "confidence": "High human-origin likelihood by source/date/attribution; not measured purity or an absolute human-only certificate",
                "limitations": policy["notes"], "outputs": {p.name: nw.digest_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
    nw.save_json(output / "manifest.json", manifest)
    print(json.dumps({"counts": dict(counts), "parent_documents": len(groups), "exclusions": dict(excluded)}, indent=2), flush=True)


def verify(output):
    manifest = read(output / "manifest.json")
    for name, expected in manifest["outputs"].items():
        if Path(name).name != name or nw.digest_file(output / name) != expected:
            raise ValueError(f"Output checksum mismatch: {name}")
    for name, expected in manifest["evidence_files"].items():
        if nw.digest_file(ROOT / name) != expected:
            raise ValueError("Evidence snapshot changed")
    source, parent_dir = Path(manifest["input_corpus"]), Path(manifest["parent_curation"])
    if nw.digest_file(source / "manifest.json") != manifest["input_manifest_sha256"] or nw.digest_file(parent_dir / "manifest.json") != manifest["parent_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    parents, hashes = checked_inputs(source, parent_dir)
    if hashes != manifest["input_files"]:
        raise ValueError("Input provenance changed")
    policy = read(output / "policy.json")
    expected, actual = {}, {}
    for split in SPLITS:
        for row in rows(source / (split + ".jsonl.gz")):
            reason, evidence = decision(row, parents, policy)
            if reason is None:
                row["human_core_selection"] = evidence
                expected[row["id"]] = row
        for row in rows(output / (split + ".jsonl.gz")):
            if row["id"] in actual or row["split"] != split:
                raise ValueError("Duplicate ID or partition mismatch")
            actual[row["id"]] = row
    if expected != actual:
        raise ValueError("Subset differs from original passages or policy selection")
    digests, partitions, parents_seen = set(), {}, set()
    counts = collections.Counter()
    for row in actual.values():
        digest = d.text_hash(d.normalized(row["text"]))
        if digest in digests:
            raise ValueError("Duplicate passage text")
        digests.add(digest)
        if partitions.setdefault(row["split_group"], row["split"]) != row["split"]:
            raise ValueError("Source group crosses partitions")
        parents_seen.add(row["parent_document_id"])
        counts["selected_spans"] += 1
        counts["selected_words"] += row["whitespace_words"]
        counts[row["split"] + "_spans"] += 1
        counts[row["split"] + "_words"] += row["whitespace_words"]
    if any(manifest["counts"].get(k, 0) != v for k, v in counts.items()) or len(parents_seen) != manifest["parent_documents"]:
        raise ValueError("Count mismatch")
    report = {"verified": True, "manifest_sha256": nw.digest_file(output / "manifest.json"),
              "verifier_sha256": nw.digest_file(Path(__file__)), "counts": dict(counts), "parent_documents": len(parents_seen),
              "checks": ["Input/output/evidence hashes", "Exact subset of original English passages", "Full-parent disclosure scan", "Source/date/attribution policy", "Word bounds and counts", "Unique text and unchanged parent partitions"],
              "scope": "Selection and artifact integrity; human-origin likelihood is inferred, not measured or individually certified"}
    nw.save_json(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--input", type=Path, default=ROOT / "data/corpus-words-v2")
    parser.add_argument("--parents", type=Path, default=ROOT / "data/curated-v2")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "data/corpus-human-core-v1")
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        verify(args.output.resolve())
