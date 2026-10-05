"""Download pinned sources and replay the exact selected human-origin core."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "releases/human-core-v1"
SPLITS = ("train", "validation", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def plain_name(name):
    require(isinstance(name, str) and name not in ("", ".", "..")
            and Path(name).name == name and "\\" not in name,
            "Expected a plain filename")
    return name


def load_release(release=RELEASE):
    lock = json.loads((release / "release.json").read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        require(digest_file(release / plain_name(name)) == expected,
                f"Release metadata changed: {name}")
    recipe = json.loads((release / "recipe.json").read_text(encoding="utf-8"))
    require(recipe["schema_version"] == 1, "Unknown recipe version")
    require(selection_hash(recipe["records"]) == recipe["selection_sha256"],
            "Recipe selection identity changed")
    return recipe, lock


def selection_hash(rows):
    keys = ("id", "parent_document_id", "split", "split_group", "text_sha256", "whitespace_words")
    identities = sorted(({k: r[k] for k in keys} for r in rows), key=lambda r: r["id"])
    data = json.dumps(identities, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text_hash(data)


def check_source(path, spec):
    require(path.stat().st_size == spec["bytes"] and digest_file(path) == spec["sha256"],
            f"Pinned source differs: {path}. Existing files were preserved; "
            "obtain the pinned version rather than changing the checksum.")


def fetch_source(spec, cache, offline=False):
    path = cache / plain_name(spec["cache_path"])
    if path.exists():
        check_source(path, spec)
        print(f"Verified cached {path.name}", flush=True)
        return path
    require(not offline, f"Offline source missing: {path}")
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}", flush=True)
    request = urllib.request.Request(spec["url"], headers={
        "User-Agent": "Mozilla/5.0 (compatible; pretraining-datawork research)"})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=path.name + ".", suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                if spec["kind"] == "gzip_jsonl_prefix":
                    with gzip.GzipFile(fileobj=response) as stream:
                        for _ in range(spec["rows"]):
                            line = stream.readline()
                            require(bool(line), "Hansard shard ended before the pinned prefix")
                            json.loads(line)
                            handle.write(line)
                else:
                    require(spec["kind"] == "file", "Unknown download kind")
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
        check_source(temporary, spec)
        # Do not replace a cache file created by another process.
        if path.exists():
            check_source(path, spec)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Verified downloaded {path.name}", flush=True)
    return path


def checked_range(value, limit):
    require(isinstance(value, list) and len(value) == 2
            and all(type(x) is int for x in value)
            and 0 <= value[0] < value[1] <= limit, "Invalid source slice")
    return value


def reconstruct_parents(recipe, cache):
    # Reuse the exact mechanical formatting operation used for the original core.
    from diverse import restore_punctuation
    import pyarrow.parquet as pq

    file_data, parents = {}, {}
    for item in recipe["parents"]:
        meta, selector = item["metadata"], item["selector"]
        name, kind = plain_name(selector["cache_path"]), selector["kind"]
        if name not in file_data:
            path = cache / name
            if kind == "wiki":
                file_data[name] = pq.read_table(path, columns=["text"])["text"].to_pylist()
            elif kind == "hansard":
                with path.open(encoding="utf-8") as handle:
                    file_data[name] = [json.loads(line) for line in handle]
            elif kind == "book":
                file_data[name] = path.read_bytes().decode("utf-8-sig")
            else:
                raise ValueError("Unknown parent source kind")
        raw = file_data[name]
        if kind == "wiki":
            a, b = checked_range(selector["range"], len(raw))
            raw_text = "".join(raw[a:b])
            require(text_hash(raw_text) == selector["raw_text_sha256"], "Raw article differs")
            body = restore_punctuation(raw_text)
        elif kind == "hansard":
            index = selector["row"]
            require(type(index) is int and 1 <= index <= len(raw), "Invalid Hansard source row")
            source = raw[index - 1]
            require(source["id"] == meta["original_id"] and source["source"] == meta["source"],
                    "Hansard source identity differs")
            a, b = checked_range(selector["range"], len(source["text"]))
            body = source["text"][a:b]
        else:
            a, b = checked_range(selector["range"], len(raw))
            body = raw[a:b]
        require(text_hash(body) == meta["text_sha256"], f"Parent body differs: {meta['id']}")
        require(len(body.split()) == meta["whitespace_words"], "Parent word count differs")
        require(meta["id"] not in parents, "Duplicate parent identity")
        parents[meta["id"]] = {**meta, "text": body}
    return parents


def reconstruct_records(recipe, parents):
    records = []
    for meta in recipe["records"]:
        body = parents[meta["parent_document_id"]]["text"]
        a, b = checked_range(meta["character_range"], len(body))
        text = body[a:b]
        require(text_hash(text) == meta["text_sha256"], f"Passage differs: {meta['id']}")
        require(len(text.split()) == meta["whitespace_words"], "Passage word count differs")
        records.append({**meta, "text": text})
    return records


def check_records(records, recipe):
    require(len(records) == recipe["passages"], "Wrong passage count")
    require(selection_hash(records) == recipe["selection_sha256"], "Wrong selected text or partitions")
    ids, texts, groups, parents = set(), set(), {}, {}
    totals = {split: {"passages": 0, "words": 0} for split in SPLITS}
    for row in records:
        text, split = row["text"], row["split"]
        words = len(text.split())
        require(split in SPLITS and row["language"] == "en", "Incorrect language or split")
        require(row["authorship"] == "human_candidate" and row["binary_target"] is None,
                "Unexpected authorship label")
        require(400 <= words <= 3000 and words == row["whitespace_words"], "Word bounds differ")
        require(text_hash(text) == row["text_sha256"], "Text checksum differs")
        require(row["id"] not in ids and row["text_sha256"] not in texts, "Duplicate passage")
        ids.add(row["id"]); texts.add(row["text_sha256"])
        require(groups.setdefault(row["split_group"], split) == split, "Group crosses partitions")
        require(parents.setdefault(row["parent_document_id"], split) == split, "Parent crosses partitions")
        totals[split]["passages"] += 1
        totals[split]["words"] += words
    require(dict(totals) == recipe["summary"]["by_split"], "Split totals differ")
    require(sum(v["words"] for v in totals.values()) == recipe["words"], "Total words differ")


def write_rows(path, records):
    # Empty filename and zero mtime make archive headers independent of location/time.
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verify(output, recipe, lock):
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["release"] == recipe["release"]
            and manifest["recipe_sha256"] == lock["files"]["recipe.json"]
            and manifest["selection_sha256"] == recipe["selection_sha256"]
            and manifest["original_manifest_sha256"] == recipe["original_manifest_sha256"],
            "Release identity differs")
    require(manifest["input_files"] == {s["cache_path"]: s["sha256"] for s in recipe["sources"]},
            "Input pins differ")
    required = {s + ".jsonl.gz" for s in SPLITS} | {
        "parent_documents.jsonl.gz", "licenses.snapshot.json", "ATTRIBUTION.txt", "diversity.json"}
    require(set(manifest["outputs"]) == required, "Unexpected output file inventory")
    for name, expected in manifest["outputs"].items():
        require(digest_file(output / plain_name(name)) == expected, f"Export changed: {name}")
    for name in ("licenses.snapshot.json", "ATTRIBUTION.txt"):
        require(manifest["outputs"][name] == lock["files"][name], "Source rights metadata differs")
    records = []
    for split in SPLITS:
        part = read_rows(output / (split + ".jsonl.gz"))
        require(all(row["split"] == split for row in part), "Record in wrong partition file")
        records.extend(part)
    check_records(records, recipe)
    parents = read_rows(output / "parent_documents.jsonl.gz")
    parent_map = {r["id"]: r for r in parents}
    require(len(parent_map) == len(parents) == recipe["parent_documents"], "Wrong parent inventory")
    require([{k: v for k, v in r.items() if k != "text"} for r in parents]
            == [p["metadata"] for p in recipe["parents"]], "Parent metadata differs")
    for r in parents:
        require(text_hash(r["text"]) == r["text_sha256"]
                and len(r["text"].split()) == r["whitespace_words"], "Changed parent body")
    require([{k: v for k, v in r.items() if k != "text"} for r in records]
            == recipe["records"], "Passage metadata or order differs")
    for r in records:
        a, b = r["character_range"]
        require(parent_map[r["parent_document_id"]]["text"][a:b] == r["text"], "Parent slice differs")
    require(json.loads((output / "diversity.json").read_text()) == recipe["summary"], "Summary differs")
    require(manifest["passages"] == recipe["passages"] and manifest["words"] == recipe["words"],
            "Manifest totals differ")
    report = {"verified": True, "manifest_sha256": digest_file(manifest_path),
              "recipe_sha256": lock["files"]["recipe.json"], "selection_sha256": recipe["selection_sha256"],
              "passages": len(records), "words": recipe["words"],
              "scope": "Exact selected text, metadata, source slices, splits and export integrity; purity and rights not certified"}
    return report


def prepare(output, cache, offline=False, verify_only=False, release=RELEASE):
    recipe, lock = load_release(release)
    if output.exists() and any(output.iterdir()):
        report = verify(output, recipe, lock)
        print(f"Already ready and verified: {output} ({report['words']:,} words)", flush=True)
        return report
    require(not verify_only, f"No prepared corpus at {output}")
    for spec in recipe["sources"]:
        fetch_source(spec, cache, offline)
    parents = reconstruct_parents(recipe, cache)
    records = reconstruct_records(recipe, parents)
    check_records(records, recipe)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", suffix=".building", dir=output.parent))
    try:
        for split in SPLITS:
            write_rows(temporary / (split + ".jsonl.gz"), (r for r in records if r["split"] == split))
        write_rows(temporary / "parent_documents.jsonl.gz", parents.values())
        for name in ("licenses.snapshot.json", "ATTRIBUTION.txt"):
            shutil.copyfile(release / name, temporary / name)
        (temporary / "diversity.json").write_bytes(json_bytes(recipe["summary"]))
        manifest = {"release": recipe["release"], "recipe_sha256": lock["files"]["recipe.json"],
                    "selection_sha256": recipe["selection_sha256"],
                    "original_manifest_sha256": recipe["original_manifest_sha256"],
                    "passages": recipe["passages"], "words": recipe["words"],
                    "summary": recipe["summary"], "input_files": {
                        s["cache_path"]: s["sha256"] for s in recipe["sources"]},
                    "processing_complete": True, "portability": recipe["notes"],
                    "outputs": {p.name: digest_file(p) for p in sorted(temporary.iterdir())}}
        (temporary / "manifest.json").write_bytes(json_bytes(manifest))
        report = verify(temporary, recipe, lock)
        (temporary / "verification.json").write_bytes(json_bytes(report))
        # Existing nonempty exports are never replaced.
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Ready: {output}\n{recipe['passages']:,} passages; {recipe['words']:,} words", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ready/human-core-v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/ready-core-v1")
    parser.add_argument("--offline", action="store_true", help="Use only already-pinned cached sources")
    parser.add_argument("--verify-only", action="store_true", help="Verify the prepared export without downloading")
    args = parser.parse_args()
    try:
        prepare(args.output.resolve(), args.cache.resolve(), args.offline, args.verify_only)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
