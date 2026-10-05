"""Acquire pinned news archives and inspect their raw document inventory."""

import argparse
import collections
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/news_downloads.json"
RAW = ROOT / "data/raw/news"
OUT = ROOT / "data/news"


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def fetch_file(spec):
    path = RAW / spec["local_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != spec["bytes"] or digest_file(path) != spec["sha256"]:
            raise ValueError(f"Existing file failed verification; not overwritten: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "pretraining-datawork/0.1"})
                digest = hashlib.sha256()
                count = 0
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                    while block := response.read(1024 * 1024):
                        digest.update(block)
                        count += len(block)
                        handle.write(block)
                if count != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
                    raise ValueError(f"Size/checksum mismatch from {spec['url']}")
                temporary.replace(path)
                break
            except ValueError:
                raise
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
            finally:
                temporary.unlink(missing_ok=True)
    return {**spec, "path": str(path), "status": "verified", "hash_scope": "complete downloaded file"}


def fetch(config):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_sha256": digest_file(CONFIG), "script_sha256": digest_file(Path(__file__)),
        "datasets": [], "access_required": config["access_required"],
        "limitations": ["Raw acquisitions are not certified human text", "No detector or authorship filter applied"],
    }
    by_name = {}
    for dataset in config["datasets"]:
        entry = {k: v for k, v in dataset.items() if k != "files"}
        entry["files"] = []
        manifest["datasets"].append(entry)
        by_name[dataset["name"]] = entry
    save_json(OUT / "acquisition.json", manifest)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch_file, spec): (dataset["name"], spec)
                   for dataset in config["datasets"] for spec in dataset["files"]}
        for future in concurrent.futures.as_completed(pending):
            name, spec = pending[future]
            try:
                result = future.result()
                print(f"Verified {spec['local_path']} ({spec['bytes']:,} bytes)", flush=True)
            except Exception as error:
                result = {**spec, "status": "failed", "error": str(error)}
                failures.append(result)
                print(f"Failed {spec['local_path']}: {error}", flush=True)
            by_name[name]["files"].append(result)
            save_json(OUT / "acquisition.json", manifest)
    for entry in manifest["datasets"]:
        entry["files"].sort(key=lambda f: f["local_path"])
    manifest["public_downloads_complete"] = not failures
    manifest["verified_bytes"] = sum(f["bytes"] for ds in manifest["datasets"] for f in ds["files"] if f["status"] == "verified")
    save_json(OUT / "acquisition.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun fetch to retry")


def inspect(config):
    import pyarrow.parquet as pq
    report = {"datasets": [], "sample_policy": "All downloaded rows inspected; raw text unchanged",
              "tokenizer": "None: length thresholds use whitespace words, not model tokens",
              "limitations": ["Word counts do not establish authorship or licensing", "Publication dates are not capture/version dates",
                              "No contamination estimate; no pretraining selection has been made"]}
    for dataset in config["datasets"]:
        stats = collections.Counter()
        domains = collections.Counter()
        licenses = collections.Counter()
        sources = collections.Counter()
        years = collections.Counter()
        fields = collections.Counter()
        files = []
        for spec in dataset["files"]:
            path = RAW / spec["local_path"]
            if path.stat().st_size != spec["bytes"] or digest_file(path) != spec["sha256"]:
                raise ValueError(f"Input failed verification: {path}")
            count = 0
            if path.suffix == ".parquet":
                parquet = pq.ParquetFile(path)
                batches = (batch.to_pylist() for batch in parquet.iter_batches(batch_size=2048))
            else:
                def json_batches():
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            yield [json.loads(line)]
                batches = json_batches()
            for batch in batches:
                for row in batch:
                    count += 1
                    stats["rows"] += 1
                    fields.update(row.keys())
                    metadata = row.get("metadata") or {}
                    url = row.get("url") or metadata.get("url") or ""
                    try:
                        domains[urlparse(url).hostname or "missing"] += 1
                    except ValueError:
                        domains["invalid"] += 1
                    sources[row.get("source") or "cc-news"] += 1
                    licenses[str(metadata.get("license") or "unspecified")] += 1
                    if dataset["name"] != "cc_news":
                        # Heterogeneous strings can include updates; do not guess a universal date.
                        stats["created_present"] += row.get("created") is not None
                        stats["author_present"] += bool(metadata.get("author"))
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        stats["empty_or_nontext"] += 1
                        continue
                    words = len(text.split())
                    stats["whitespace_words"] += words
                    stats["text_utf8_bytes"] += len(text.encode("utf-8"))
                    for cutoff in (100, 400, 1000):
                        if words >= cutoff:
                            stats[f"rows_at_least_{cutoff}_words"] += 1
                            stats[f"words_in_rows_at_least_{cutoff}_words"] += words
                    if dataset["name"] == "cc_news":
                        try:
                            publication = dt.datetime.fromisoformat(row["date"])
                            years[str(publication.year)] += 1
                            stats["publication_through_2021"] += publication.year <= 2021
                        except (ValueError, TypeError, KeyError):
                            stats["missing_or_invalid_publication_date"] += 1
            if path.suffix == ".parquet" and count != parquet.metadata.num_rows:
                raise ValueError(f"Parquet row-count mismatch: {path}")
            files.append({"path": str(path), "sha256": spec["sha256"], "rows": count})
            print(f"Inspected {spec['local_path']}: {count:,} rows", flush=True)
        report["datasets"].append({"name": dataset["name"], "repo": dataset["repo"], "revision": dataset["revision"],
                                   "counts": dict(stats), "top_url_hosts": domains.most_common(30), "sources": dict(sources),
                                   "license_values": dict(licenses), "publication_years": dict(years), "field_coverage": dict(fields),
                                   "files": files})
    report["config_sha256"] = digest_file(CONFIG)
    report["script_sha256"] = digest_file(Path(__file__))
    report["pyarrow_version"] = __import__("pyarrow").__version__
    save_json(OUT / "inspection.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "inspect"])
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    {"fetch": fetch, "inspect": inspect}[args.command](config)


if __name__ == "__main__":
    main()
