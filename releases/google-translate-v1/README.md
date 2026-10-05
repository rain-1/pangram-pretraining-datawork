# Frozen Google Translate controls

This selection describes **3,327 long English detector controls** from the official Par3 release: 1,055 Google Translate paragraphs and 2,272 human translations. All are intact paragraphs with 400–3,000 words. Human translations are valid negatives for this project, including texts written in other languages and translated into English by people.

Run `uv run --frozen python prepare_translations.py` from the repository root. The command downloads the pinned 285 MB archive, rejects executable pickle construction, extracts the original selection and verifies each English/source text hash and process label. Ready data is `data/ready/google-translate-v1/binary_examples.jsonl`; read `text`, `binary_target` (1 machine, 0 human) and `split`. No model is trained or used to filter the pretraining core.

| Partition | Google Translate | Human translations | Total |
| --- | ---: | ---: | ---: |
| Train | 969 | 2,089 | 3,058 |
| Validation | 68 | 140 | 208 |
| Test | 18 | 43 | 61 |

`release.json` pins the source configuration, selection metadata and licence snapshot. `selection.json` contains IDs, English and foreign-source hashes, process labels, alignment positions and existing work-level splits; it contains no passage prose. Selection identity is SHA-256 over compact, key-sorted UTF-8 JSON of the identities sorted by ID, implemented by `prepare_translations.identities` and `identity_hash`:

```text
4e518c3d927f7571f1376c9a9024817e630b6bd3e02882c91544ce44e585ebb5
```

The original local `data/detector-par3-v2/` manifest SHA-256 is `70c5b964ee72d3e20e2df0711ddc4acdaf67f7d3c3003ede46768b25d6f7747c`. Portable exports retain the same text, IDs, labels, aligned source paragraphs and partitions, with relative acquisition paths and their own generated manifest/checksums. `records.jsonl` overlaps the binary export and is not extra training data.

These literary controls span 12 source languages and 69 connected work groups. The small test split and uneven languages/authors limit generalization claims. Original paragraph order is shuffled, so adjacent release rows must not be joined. Individual literary text/translation rights, exact Google versions and generation dates remain unverified; the upstream repository's MIT licence does not settle text rights. Keep these controls separate from pretraining data. The existing baseline's 71.3% human false-positive rate makes it unsuitable for corpus removal.
