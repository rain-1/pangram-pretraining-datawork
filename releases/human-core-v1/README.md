# Frozen human core selection

This release recipe reconstructs the exact selected text and partitions from the project's diverse human-origin core: **463 passages, 1,109,431 whitespace words, 268 parent documents**. It contains source metadata, locations, offsets and hashes, not passage prose. The public text is downloaded from the pinned upstream sources by `prepare.py`.

`release.json` pins `recipe.json`, `licenses.snapshot.json` and `ATTRIBUTION.txt`. `recipe.json` pins raw source bytes and each parent/passage body; it retains source-specific authorship evidence and uncertainty. Replay preserves the original selection instead of rerunning selection against a changing pool. Frozen parent and passage offsets apply to the exact pinned input, including the original mechanical WikiText punctuation restoration.

The original local core manifest SHA-256 is:

```text
2f8a39152727a7bfb589d5b193df787c0ebc6038b0a9afe7023ff14667958ba6
```

The portable selection SHA-256 is:

```text
0bacc68d52fd847503c581131f88180edde3e9254525bc8655c063f50cab1f05
```

Selection identity hashes the UTF-8 canonical JSON of the sorted passage identity records: `id`, `parent_document_id`, `split`, `split_group`, `text_sha256` and `whitespace_words`, sorted by `id`, with sorted object keys and compact separators. The exact rule is implemented by `prepare.selection_hash`.

Paths in record metadata are relative to the original workspace layout. They describe acquisition provenance; setup's current cache can be elsewhere. Gzip headers use zero timestamps and omit filenames. Portable manifest/archive bytes differ from the original local export. A verification report is generated against the local portable files after replay.

Use `uv run --frozen python prepare.py` from the repository root. For details and loading code, read [DATA_HANDOFF.md](../../DATA_HANDOFF.md). Source licences, attribution requirements and unresolved issues are preserved in this directory. Repository publication does not certify individual authorship, measure contamination or grant additional rights to upstream text.
