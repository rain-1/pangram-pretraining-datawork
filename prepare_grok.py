"""Index pinned GrokSet IDs, then optionally import locally hydrated original text.

This command never calls X or the authors' hydration toolkit.
"""

import argparse
import collections
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import ijson

import datawork as d
import prepare as p
import prepare_twitter as pt
import twitterwork as tw

ROOT = Path(__file__).resolve().parent


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def tweet_id(value):
    p.require(isinstance(value, str) and value.isascii() and value.isdigit(), "Invalid tweet ID")
    return value


def flag(value):
    p.require(value is None or type(value) is bool, "Unexpected metadata boolean")
    return None if value is None else int(value)


def connect_readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def index_metadata(source, destination):
    """Stream the top-level array; discard all annotations, including LLM labels."""
    groups = Groups()
    counts = collections.Counter()
    with sqlite3.connect(destination / "index.sqlite") as db:
        db.executescript("""
            CREATE TABLE tweets(id TEXT PRIMARY KEY, is_assistant INTEGER, lang TEXT,
              reply_to TEXT, media_only INTEGER, created_at TEXT, conflict INTEGER NOT NULL);
            CREATE TABLE memberships(tweet_id TEXT, conversation_id TEXT, thread_id TEXT,
              PRIMARY KEY(tweet_id, conversation_id, thread_id));
            CREATE TABLE conversation_groups(conversation_id TEXT PRIMARY KEY, split_group TEXT);
        """)
        with source.open("rb") as handle:
            for conversation in ijson.items(handle, "item"):
                cid = tweet_id(conversation["conversationId"])
                groups.find(cid)
                counts["conversations"] += 1
                p.require(isinstance(conversation["threads"], list), "Invalid Grok threads")
                for thread in conversation["threads"]:
                    tid = tweet_id(thread["threadId"])
                    counts["threads"] += 1
                    for item in thread["tweets"]:
                        ident = tweet_id(item["id"])
                        author = item.get("author") or {}
                        p.require(isinstance(author, dict), "Invalid Grok author metadata")
                        role, media = flag(author.get("isAssistant")), flag(item.get("isMediaOnly"))
                        lang = item.get("lang")
                        p.require(lang is None or isinstance(lang, str), "Invalid Grok language")
                        reply = item.get("inReplyToId") or None
                        if reply is not None:
                            reply = tweet_id(reply)
                        stamp = item.get("createdAt")
                        p.require(stamp is None or isinstance(stamp, str), "Invalid Grok date")
                        values = (role, lang, reply, media, stamp)
                        old = db.execute("SELECT is_assistant,lang,reply_to,media_only,created_at FROM tweets WHERE id=?",
                                         (ident,)).fetchone()
                        if old is None:
                            db.execute("INSERT INTO tweets VALUES(?,?,?,?,?,?,0)", (ident, *values))
                        elif old != values:
                            db.execute("UPDATE tweets SET conflict=1 WHERE id=?", (ident,))
                        existing = db.execute("SELECT conversation_id FROM memberships WHERE tweet_id=? LIMIT 1",
                                              (ident,)).fetchone()
                        if existing:
                            groups.union(cid, existing[0])
                        db.execute("INSERT OR IGNORE INTO memberships VALUES(?,?,?)", (ident, cid, tid))
                        counts["tweet_occurrences"] += 1
                if counts["conversations"] % 20000 == 0:
                    db.commit()
                    print(f"Indexed {counts['conversations']:,} conversations", flush=True)
        db.executemany("INSERT INTO conversation_groups VALUES(?,?)",
                       ((cid, p.text_hash("grok-conversation:" + groups.find(cid))) for cid in sorted(groups.parent)))
        db.commit()
        counts["unique_tweet_ids"] = db.execute("SELECT count(*) FROM tweets").fetchone()[0]
        counts["conflicting_metadata_ids"] = db.execute("SELECT count(*) FROM tweets WHERE conflict=1").fetchone()[0]
        counts["connected_conversation_groups"] = len({groups.find(cid) for cid in groups.parent})
        query = """SELECT t.id,t.reply_to,t.created_at,g.split_group
                   FROM tweets t JOIN memberships m ON m.tweet_id=t.id
                   JOIN conversation_groups g ON g.conversation_id=m.conversation_id
                   WHERE t.is_assistant=1 AND t.lang='en' AND t.reply_to IS NOT NULL
                   AND t.media_only=0 AND t.conflict=0 GROUP BY t.id ORDER BY t.id"""
        candidates = (dict(tweet_id=ident, in_reply_to_id=reply, created_at=stamp, split_group=group,
                           candidate_role="assistant_reply", language="en", text_available=False,
                           binary_target=None) for ident, reply, stamp, group in db.execute(query))
        p.write_rows(destination / "assistant_candidates.jsonl.gz", candidates)
        counts["english_assistant_reply_ids"] = db.execute("""SELECT count(*) FROM tweets WHERE
            is_assistant=1 AND lang='en' AND reply_to IS NOT NULL AND media_only=0 AND conflict=0""").fetchone()[0]
    return dict(counts)


def verify_metadata(output, spec, lock):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    p.require(manifest["kind"] == "grok_ids_only" and manifest["source_sha256"] == spec["sha256"]
              and manifest["release_files"] == lock["files"], "Grok metadata release differs")
    p.require(set(manifest["outputs"]) == {"index.sqlite", "assistant_candidates.jsonl.gz", "licenses.snapshot.json"},
              "Unexpected Grok metadata inventory")
    for name, digest in manifest["outputs"].items():
        p.require(p.digest_file(output / p.plain_name(name)) == digest, f"Grok metadata changed: {name}")
    p.require(manifest["outputs"]["licenses.snapshot.json"] == lock["files"]["licenses.snapshot.json"],
              "Grok rights inventory differs")
    with connect_readonly(output / "index.sqlite") as db:
        p.require(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Grok SQLite integrity failure")
        p.require(db.execute("SELECT count(*) FROM tweets").fetchone()[0] == manifest["counts"]["unique_tweet_ids"],
                  "Grok index count differs")
    return manifest


def publish(temporary, output):
    if output.exists():
        output.rmdir()
    temporary.rename(output)


def metadata(output, cache, offline=False, verify_only=False):
    _, config, lock = pt.load_release()
    spec = next(s for s in config["sources"] if s["cache_path"] == "grokset-dehydrated.json")
    if output.exists() and any(output.iterdir()):
        return verify_metadata(output, spec, lock)
    p.require(not verify_only, f"No prepared Grok IDs at {output}")
    source = p.fetch_source(spec, cache, offline)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", suffix=".building", dir=output.parent))
    try:
        counts = index_metadata(source, temporary)
        shutil.copyfile(pt.RELEASE / "licenses.snapshot.json", temporary / "licenses.snapshot.json")
        manifest = {"kind": "grok_ids_only", "source_sha256": spec["sha256"],
                    "source_revision": spec["revision"], "release_files": lock["files"], "counts": counts,
                    "outputs": {f.name: p.digest_file(f) for f in sorted(temporary.iterdir())}}
        (temporary / "manifest.json").write_bytes(p.json_bytes(manifest))
        verify_metadata(temporary, spec, lock)
        publish(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def read_hydrated(path, db):
    """Accept only flat JSONL {id, original_text, lang?}; no guessed text fields."""
    rows, rejected, seen = [], [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            p.require(isinstance(item, dict) and set(item) <= {"id", "original_text", "lang"}
                      and {"id", "original_text"} <= set(item),
                      "Hydrated JSONL requires id and original_text, optionally lang; export raw original text explicitly")
            ident = tweet_id(item["id"])
            p.require(ident not in seen, "Repeated hydrated tweet ID")
            seen.add(ident)
            text = item["original_text"]
            p.require(isinstance(text, str), "Hydrated original_text must be a string")
            meta = db.execute("""SELECT t.is_assistant,t.lang,t.reply_to,t.media_only,t.created_at,t.conflict,g.split_group
                FROM tweets t JOIN memberships m ON m.tweet_id=t.id
                JOIN conversation_groups g ON g.conversation_id=m.conversation_id WHERE t.id=? LIMIT 1""", (ident,)).fetchone()
            if not meta:
                rejected.append({"line": number, "tweet_id": ident, "reason": "id_not_in_pinned_release"})
                continue
            role, lang, reply, media, stamp, conflict, group = meta
            reason = None
            if conflict:
                reason = "conflicting_release_metadata"
            elif role != 1:
                reason = "participant_authorship_unresolved"
            elif lang != "en" or item.get("lang", "en") != "en":
                reason = "not_english_or_language_conflict"
            elif reply is None or media != 0:
                reason = "not_a_text_assistant_reply"
            elif not d.normalized(text):
                reason = "empty_original_text"
            elif text.lstrip().startswith("RT @"):
                reason = "repost_text"
            if reason:
                rejected.append({"line": number, "tweet_id": ident, "reason": reason})
                continue
            rows.append({"id": "grok:" + ident, "tweet_id": ident, "text": text,
                         "text_sha256": p.text_hash(text), "normalized_text_sha256": p.text_hash(d.normalized(text)),
                         "binary_target": 1, "authorship": "assistant_account_reply", "engine": "Grok (version unknown)",
                         "label_basis": "Pinned author.isAssistant flag; local original text not independently verified",
                         "human_authorship_verified": False, "intended_use": "twitter_detector_research_only",
                         "language": "en", "in_reply_to_id": reply, "created_at": stamp,
                         "split_group": group, "quoted_or_copied_human_tokens_possible": True})
    # Conversation components are established on ALL released IDs, not only
    # hydrated ones. Also merge components sharing normalized original text.
    groups, first = Groups(), {}
    for row in rows:
        groups.union(row["split_group"], first.setdefault(row["normalized_text_sha256"], row["split_group"]))
    canonical = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        row["split_group"] = groups.find(row["split_group"])
        bucket = int(p.text_hash("grok-split:" + row["split_group"])[:8], 16) % 100
        row["split"] = "test" if bucket < 10 else ("validation" if bucket < 20 else "train")
        key = row["normalized_text_sha256"]
        if key not in canonical:
            canonical[key] = {**row, "equivalent_record_ids": []}
        canonical[key]["equivalent_record_ids"].append(row["id"])
    return rows, sorted(canonical.values(), key=lambda r: r["id"]), rejected


def import_text(input_path, metadata_path, output):
    _, config, lock = pt.load_release()
    spec = next(s for s in config["sources"] if s["cache_path"] == "grokset-dehydrated.json")
    verify_metadata(metadata_path, spec, lock)
    input_digest = p.digest_file(input_path)
    metadata_digest = p.digest_file(metadata_path / "manifest.json")
    if output.exists() and any(output.iterdir()):
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        p.require(manifest["kind"] == "grok_local_hydrated_text"
                  and manifest["input_sha256"] == input_digest
                  and manifest["metadata_manifest_sha256"] == metadata_digest, "Existing Grok import has different inputs")
        expected = {"records.jsonl.gz", "rejected.json", "licenses.snapshot.json"}
        expected.update(split + ".jsonl.gz" for split in p.SPLITS)
        p.require(set(manifest["outputs"]) == expected, "Unexpected Grok import inventory")
        for name, digest in manifest["outputs"].items():
            p.require(p.digest_file(output / p.plain_name(name)) == digest, f"Grok import changed: {name}")
        return manifest
    with connect_readonly(metadata_path / "index.sqlite") as db:
        rows, binary, rejected = read_hydrated(input_path, db)
    p.require(p.digest_file(input_path) == input_digest, "Hydrated input changed during import")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", suffix=".building", dir=output.parent))
    try:
        p.write_rows(temporary / "records.jsonl.gz", rows)
        for split in p.SPLITS:
            p.write_rows(temporary / (split + ".jsonl.gz"), [r for r in binary if r["split"] == split])
        (temporary / "rejected.json").write_bytes(p.json_bytes(rejected))
        shutil.copyfile(pt.RELEASE / "licenses.snapshot.json", temporary / "licenses.snapshot.json")
        manifest = {"kind": "grok_local_hydrated_text", "input_sha256": input_digest,
                    "metadata_manifest_sha256": metadata_digest, "release_files": lock["files"],
                    "audit_records": len(rows), "binary_examples": len(binary), "rejected_rows": len(rejected),
                    "selection_sha256": tw.stable_hash(binary),
                    "by_split": dict(collections.Counter(r["split"] for r in binary)),
                    "warning": "Assistant positives only; requires paired representative human references and cross-source deduplication before evaluation",
                    "outputs": {f.name: p.digest_file(f) for f in sorted(temporary.iterdir())}}
        (temporary / "manifest.json").write_bytes(p.json_bytes(manifest))
        publish(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("metadata", help="Download ~1.05 GB of pinned IDs/metadata; no tweet text")
    ids.add_argument("--cache", type=Path, default=ROOT / "data/raw/twitter")
    ids.add_argument("--output", type=Path, default=ROOT / "data/ready/grok-ids-v1")
    ids.add_argument("--offline", action="store_true")
    ids.add_argument("--verify-only", action="store_true")
    hydrated = sub.add_parser("import-text", help="Import an existing local original-text JSONL export")
    hydrated.add_argument("--input", type=Path, required=True)
    hydrated.add_argument("--metadata", type=Path, default=ROOT / "data/ready/grok-ids-v1")
    hydrated.add_argument("--output", type=Path, default=ROOT / "data/ready/grok-text-v1")
    args = parser.parse_args()
    try:
        if args.command == "metadata":
            result = metadata(args.output.resolve(), args.cache.resolve(), args.offline, args.verify_only)
        else:
            result = import_text(args.input.resolve(), args.metadata.resolve(), args.output.resolve())
        print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, indent=2))
    except (ValueError, OSError, sqlite3.Error, ijson.JSONError) as exc:
        parser.exit(1, f"Grok preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
