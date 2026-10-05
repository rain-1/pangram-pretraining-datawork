"""Stream local corpora into a provenance-preserving, token-budgeted selection."""

import argparse
import collections
import contextlib
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET

from tokenizers import Tokenizer
import datawork as d

PILE_ALLOWLIST = {"FreeLaw", "NIH ExPorter", "PubMed Central", "PubMed Abstracts",
                  "ArXiv", "PhilPapers", "StackExchange", "Wikipedia (en)"}


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def files_for(path):
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input does not exist: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file() and (p.suffix in {".gz", ".jsonl", ".sgm", ".sgml"}))
    if not files:
        raise ValueError(f"No input .gz/.jsonl/.sgm/.sgml files under {path}")
    return files


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def gigaword_documents(path):
    """LDC Gigaword has one SGML tag per line, complete <DOC> blocks."""
    with open_text(path) as handle:
        block, start = None, None
        for line_number, line in enumerate(handle, 1):
            if line.lstrip().startswith("<DOC "):
                if block is not None:
                    raise ValueError(f"Nested/incomplete DOC at {path}:{line_number}")
                block, start = [line], line_number
            elif block is not None:
                block.append(line)
            if block is not None and line.strip() == "</DOC>":
                yield ET.fromstring("".join(block)), start
                block = None
        if block is not None:
            raise ValueError(f"Unclosed DOC at {path}:{start}")


def gigaword_rows(path, cutoff):
    for doc, line in gigaword_documents(path):
        key = doc.attrib.get("id", f"line-{line}")
        reason = None
        match = re.fullmatch(r"([A-Z]+_ENG)_(\d{8})\.(\d+)", key)
        date = None
        if doc.attrib.get("type") != "story":
            reason = "non_story"
        elif match is None:
            reason = "invalid_story_id"
        else:
            try:
                date = dt.datetime.strptime(match[2], "%Y%m%d").date()
            except ValueError:
                reason = "invalid_story_date"
            if date is not None and date > cutoff:
                reason = "publication_after_cutoff"
        body = doc.find("TEXT")
        if body is None and reason is None:
            reason = "missing_story_text"
        if reason:
            yield None, {"id": key, "line": line, "reason": reason}
            continue
        paragraphs = body.findall("P")
        text = "\n\n".join("".join(p.itertext()).strip() for p in paragraphs) if paragraphs else "".join(body.itertext()).strip()
        if not text:
            yield None, {"id": key, "line": line, "reason": "empty_story_text"}
            continue
        yield d.record("gigaword", key, text, "documented_human", key,
                       publication_date=date.isoformat(), source=match[1],
                       evidence="LDC2011T07 README describes manually generated source data; story category is heuristic",
                       version_evidence="Archived LDC release; document date from DOC id; exact revision timestamp unavailable",
                       license_status="User-supplied licensed LDC corpus; permissions depend on the user's agreement"), None


def jsonl_rows(path, adapter, cutoff, snapshot_date):
    with open_text(path) as handle:
        for line, text in enumerate(handle, 1):
            row = json.loads(text)
            if adapter == "hansard":
                reason, date = d.hansard_decision(row, cutoff.year)
                if reason == "retain_candidate" and dt.date.fromisoformat(date) > cutoff:
                    reason = "publication_after_cutoff"
                if reason != "retain_candidate":
                    yield None, {"id": row.get("id"), "line": line, "reason": reason}
                    continue
                record = d.record("hansard", f"{row['source']}/{row['id']}", row["text"], "human_candidate", row["id"],
                                  publication_date=date, source=row["source"], metadata=row["metadata"],
                                  collected_date=row.get("added"),
                                  evidence="Official proceedings; date verified against ID and metadata.year",
                                  version_evidence="Publication date only; captured revisions not independently verified",
                                  license_status="Open Parliament Licence reported upstream")
            else:
                component = row.get("meta", {}).get("pile_set_name")
                if component not in PILE_ALLOWLIST:
                    yield None, {"id": f"line-{line}", "line": line, "reason": "pile_component_not_allowlisted", "component": component}
                    continue
                if snapshot_date > cutoff:
                    yield None, {"id": f"line-{line}", "line": line, "reason": "snapshot_after_cutoff"}
                    continue
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    yield None, {"id": f"line-{line}", "line": line, "reason": "empty_text"}
                    continue
                record = d.record("pile", f"{component}/{d.text_hash(row['text'])}", row["text"], "human_candidate",
                                  d.text_hash(row["text"]), source=component, snapshot_date=snapshot_date.isoformat(),
                                  evidence="User-identified original Pile snapshot and allowlisted component; authorship inferred",
                                  version_evidence="Snapshot date is user-supplied and must be checked against the actual archive",
                                  license_status="Component-specific; source label does not grant a license")
            record["input_line"] = line
            yield record, None


def parse_mapping(items, allowed):
    parsed = {}
    for item in items:
        name, value = item.split("=", 1)
        if name not in allowed or name in parsed:
            raise ValueError(f"Unknown/duplicate source {name}")
        parsed[name] = value
    return parsed


def select(args):
    if args.token_budget <= 0 or args.min_document_tokens <= 0:
        raise ValueError("Token budget and minimum document size must be positive")
    inputs = parse_mapping(args.input, {"gigaword", "hansard", "pile"})
    quotas = {k: int(v) for k, v in parse_mapping(args.quota, set(inputs)).items()}
    if len(inputs) == 1 and not quotas:
        quotas = {next(iter(inputs)): args.token_budget}
    if set(quotas) != set(inputs) or sum(quotas.values()) != args.token_budget or any(v <= 0 for v in quotas.values()):
        raise ValueError("Supply a positive --quota SOURCE=TOKENS for each input, summing to --token-budget")
    if args.evidence == "documented" and set(inputs) != {"gigaword"}:
        raise ValueError("Documented mode supports Gigaword. Use provenance mode to include inferred Hansard/Pile candidates")
    cutoff = dt.date.fromisoformat(args.cutoff)
    snapshot_date = dt.date.fromisoformat(args.pile_snapshot_date) if args.pile_snapshot_date else None
    if "pile" in inputs and snapshot_date is None:
        raise ValueError("Pile requires --pile-snapshot-date for the actual local archive")
    input_files = {name: files_for(Path(path).resolve()) for name, path in inputs.items()}
    tokenizer_path = args.tokenizer_json.resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if args.eos_token_id is not None and tokenizer.id_to_token(args.eos_token_id) is None:
        raise ValueError("EOS token ID does not exist in tokenizer vocabulary")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing corpus will not be overwritten")
    for paths in input_files.values():
        for path in paths:
            if output == path or output in path.parents:
                raise ValueError("Output must be separate from input files")
    output.mkdir(parents=True, exist_ok=True)
    tokens, docs = collections.Counter(), collections.Counter()
    rejections = collections.Counter()
    manifests = []
    # Disk-backed exact deduplication avoids keeping a corpus-sized hash set in RAM.
    with contextlib.ExitStack() as stack:
        database = stack.enter_context(sqlite3.connect(output / "dedup.sqlite"))
        database.execute("CREATE TABLE seen (id TEXT UNIQUE, digest TEXT UNIQUE)")
        handles = {split: stack.enter_context((output / f"{split}.jsonl").open("w", encoding="utf-8"))
                   for split in ("train", "validation", "test")}
        rejects = stack.enter_context((output / "rejections.jsonl").open("w", encoding="utf-8"))
        def reject(adapter, path, rejection):
            reason = rejection["reason"]
            rejections[f"{adapter}:{reason}"] += 1
            rejects.write(json.dumps({"adapter": adapter, "file": str(path), **rejection}, ensure_ascii=False) + "\n")
        for adapter, paths in input_files.items():
            used = 0
            for path in paths:
                if quotas[adapter] - used < args.min_document_tokens:
                    break
                stat_before = path.stat()
                checksum = file_hash(path)
                rows = gigaword_rows(path, cutoff) if adapter == "gigaword" else jsonl_rows(path, adapter, cutoff, snapshot_date)
                scanned = 0
                complete = True
                try:
                    for row, rejection in rows:
                        scanned += 1
                        if rejection:
                            reject(adapter, path, rejection)
                            continue
                        digest = d.text_hash(d.normalized(row["text"]))
                        count = len(tokenizer.encode(row["text"], add_special_tokens=False).ids)
                        count += int(args.eos_token_id is not None)
                        reason = "document_too_short" if count < args.min_document_tokens else "document_exceeds_remaining_quota" if count > quotas[adapter] - used else None
                        if reason:
                            reject(adapter, path, {"id": row["id"], "reason": reason, "tokens": count})
                            continue
                        try:
                            database.execute("INSERT INTO seen VALUES (?, ?)", (row["id"], digest))
                        except sqlite3.IntegrityError:
                            reject(adapter, path, {"id": row["id"], "reason": "duplicate_id_or_normalized_text"})
                            continue
                        bucket = int(d.text_hash(args.seed + ":" + digest)[:16], 16) % 100
                        split = "train" if bucket < 98 else "validation" if bucket == 98 else "test"
                        row.update(token_count=count, split=split, input_file=str(path), input_file_sha256=checksum)
                        handles[split].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        tokens[adapter] += count
                        tokens[f"split:{split}"] += count
                        docs[adapter] += 1
                        used += count
                        if scanned % 10000 == 0:
                            database.commit()
                            print(f"{adapter}: {used:,}/{quotas[adapter]:,} tokens", flush=True)
                        if quotas[adapter] - used < args.min_document_tokens:
                            complete = False
                            break
                finally:
                    rows.close()
                stat_after = path.stat()
                if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
                    raise ValueError(f"Input changed during selection: {path}")
                manifests.append({"adapter": adapter, "file": str(path), "sha256": checksum,
                                  "bytes": stat_before.st_size, "scanned_documents": scanned, "reached_eof": complete})
            database.commit()
    selected = sum(tokens[k] for k in inputs)
    manifest = {"processing_complete": True, "budget_met_exactly": selected == args.token_budget,
                "requested_tokens": args.token_budget, "selected_tokens": selected,
                "unfilled_tokens": args.token_budget - selected, "quotas": quotas,
                "tokens_by_source_and_split": dict(tokens), "documents_by_source": dict(docs),
                "rejections": dict(rejections), "input_files": manifests,
                "tokenizer_file": str(tokenizer_path), "tokenizer_sha256": file_hash(tokenizer_path),
                "eos_token_id": args.eos_token_id, "special_tokens": "Only optional one EOS per selected document",
                "cutoff": args.cutoff, "evidence_mode": args.evidence, "seed": args.seed,
                "pile_allowlist": sorted(PILE_ALLOWLIST), "script_sha256": file_hash(Path(__file__)),
                "tokenizers_version": __import__("tokenizers").__version__,
                "budget_scope": "Total corpus tokens across train/validation/test; full documents; no truncation",
                "limitations": ["Source provenance is evidence, not a human-authorship certificate",
                                "No MT/LLM detector has been calibrated or applied",
                                "Input order is deterministic; selection is not a uniform random sample",
                                "Exact duplicates removed; near-duplicate/version/quoted-synthetic-text review remains",
                                "Validation/test are hash splits; publisher/topic/author independence is not established"]}
    manifest["output_sha256"] = {f"{split}.jsonl": file_hash(output / f"{split}.jsonl") for split in handles}
    d.write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="gigaword=/path, hansard=/path, or pile=/path; repeat per adapter")
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, required=True, help="Total selected tokens across all three splits")
    parser.add_argument("--quota", action="append", default=[], help="ADAPTER=TOKENS; required for multiple input adapters")
    parser.add_argument("--cutoff", default="2021-12-31")
    parser.add_argument("--evidence", choices=["documented", "provenance"], default="documented")
    parser.add_argument("--pile-snapshot-date")
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--min-document-tokens", type=int, default=512)
    parser.add_argument("--seed", default="corpus-v1")
    parser.add_argument("--output", type=Path, required=True)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
