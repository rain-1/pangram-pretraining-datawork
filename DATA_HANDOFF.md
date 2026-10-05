# Using the human origin corpus

This is the entry point for an AI or developer using the exact saved data from this project. Start with the recommended diverse core below. Its human origin is inferred from provenance; contamination has not been measured, and its candidate labels are not confirmed human labels. No learned authorship detector was used to select this core.

## Setup from GitHub

With uv installed, clone and run:

```bash
git clone https://github.com/rain-1/pangram-pretraining-datawork.git
cd pangram-pretraining-datawork
uv run --frozen python prepare.py
```

This prepares **`data/ready/human-core-v1/`**, containing the exact selected 1,109,431 whitespace words in 463 English passages of 400–3,000 words. Paths below are relative to the repository root, wherever you cloned it. The command downloads only the pinned core inputs, reconstructs the frozen selection and verifies it; it does not download the large news pool or translation controls. Repeat the command to verify an existing export without fetching again. Changed upstream files fail their checksum rather than changing the dataset.

## Exact release and files

| File | Passages | Words | Use |
| --- | ---: | ---: | --- |
| `train.jsonl.gz` | 404 | 969,765 | Training input |
| `validation.jsonl.gz` | 23 | 51,977 | Development and tuning |
| `test.jsonl.gz` | 36 | 87,689 | Final held-out evaluation |

These are gzip-compressed UTF-8 JSON Lines files. Each line is one record. **Use only `record["text"]` as model input.** Keep IDs, source metadata and partition assignments in a separate audit record. Never train on validation or test when reporting evaluation on those partitions. Small held-out sets and uneven genre coverage limit evaluation conclusions.

The core combines Gutenberg books (516,380 words), WikiText-2 reference articles (444,959 words) and Common Pile Hansard written Q&A (148,092 words). It includes fiction, nonfiction, reference and parliamentary prose. It is a small corpus with limited contemporary everyday writing, not a representative sample of all English.

The selected passage identity, including hashes and partitions, has SHA-256:

```text
0bacc68d52fd847503c581131f88180edde3e9254525bc8655c063f50cab1f05
```

The [frozen recipe](releases/human-core-v1/README.md) defines how this identity is computed and pins source bytes and metadata. The generated `manifest.json` pins every local export by checksum, and `verification.json` records the completed checks. `diversity.json` describes the mix. `parent_documents.jsonl.gz` is an audit export containing overlapping full documents; do not add it to passage training inputs.

The original workspace also preserves `data/corpus-human-diverse-core-v1/`, whose manifest SHA-256 is `2f8a39152727a7bfb589d5b193df787c0ebc6038b0a9afe7023ff14667958ba6`. The portable version has identical text, membership, order and partitions, with relative provenance paths and deterministic archive headers. Its manifest/archive checksums consequently differ from the original. A clone starts without either export until setup runs.

## Load and verify with Python

After setup, run this from the repository root. The loader and `prepare` verification helper use only Python's standard library. Verification checks the frozen recipe, export hashes, exact selected text and metadata, parent slices and split totals before streaming training text.

```python
from pathlib import Path
import gzip
import json
from prepare import load_release, verify

ROOT = Path("data/ready/human-core-v1")
recipe, release_lock = load_release()
verification = verify(ROOT, recipe, release_lock)
print("Verified", verification["passages"], "passages", verification["words"], "words")

def records(split):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    with gzip.open(ROOT / f"{split}.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)

training_texts = (record["text"] for record in records("train"))
# Pass training_texts to your tokenizer or training input pipeline.
# There is no fixed tokenizer or model-specific token budget for this release.
```

Record metadata includes `id`, `parent_document_id`, `split`, `split_group`, `dataset`, `genre`, `text_sha256`, `whitespace_words`, `license` and `human_origin_basis`. Source-specific fields vary. Preserve them when creating derived datasets. Do not change `authorship="human_candidate"` or null `binary_target` into a confirmed-human claim. Avoid using publisher, title, license or provenance fields as classifier features.

## Other available collections

The translation controls have a separate portable setup:

```bash
uv run --frozen python prepare_translations.py
```

This downloads the pinned 285 MB official Par3 archive and prepares `data/ready/google-translate-v1/binary_examples.jsonl` (plain UTF-8 JSON Lines). Use `text` as English input, `binary_target` as the label (1 machine, 0 human), and the saved `split`. Train/validation/test contain 3,058 / 208 / 61 examples. `records.jsonl` is an overlapping provenance export, not extra examples. Rerunning verifies the prepared export; `--verify-only` checks it without downloads. The code verifies native paragraph identity and disables executable pickle construction. See the [frozen selection](releases/google-translate-v1/README.md).

The legacy collections below exist in the original workspace; neither portable setup prepares these legacy directories. Their acquisition and reproduction commands are documented in the README.

| Directory | Size | Purpose |
| --- | --- | --- |
| `data/corpus-human-diverse-v1/` | 1,479,980 words, 931 passages | The core plus selected named-author journalism; weaker captured-version evidence |
| `data/corpus-words-v2/` | 45,507,097 words, 52,186 passages | Larger news and parliamentary candidate pool; weaker authorship evidence |
| `data/corpus-human-core-v1/` | 221,582 words, 104 passages | Earlier, narrowly parliamentary subset |
| `data/detector-par3-v2/` | 3,327 examples | Translation-detector research controls, kept outside pretraining |

Choose one pretraining export. These releases share text: do not concatenate them to increase apparent unique volume. Preserve existing split groups. If combining with other data, check overlap and regroup before partitioning. Source dates describe different things by dataset; do not interpret every date field as the date of the captured wording.

The portable translation controls reproduce the legacy Par3 selection: 1,055 Google Translate outputs and 2,272 human translations. The English input is `text`, not `source_text`. Do not concatenate adjacent paragraphs: release order is shuffled. Human translations are valid negatives under this project's definition. Keep all these controls outside the pretraining inputs.

The frozen model in `data/detector-v1/` falsely flagged 71.3% of Par3 human translations. **Do not use it for corpus removal.** These literary controls do not establish filtering performance on general English or corpus contamination. Generator versions, generation dates and individual translation rights remain unverified.

## Licences and portability

Read the selected release's `licenses.snapshot.json` and `ATTRIBUTION.txt`, plus the live inventory at `config/licenses.json`. Licences are tracked, not universally cleared. WikiText has conflicting declared licence versions; Gutenberg's USA public-domain declarations do not settle every jurisdiction; news rights include unresolved cases. The inventory identifies raw and derived files affected by removing a source.

The simplest way to use the core on another machine is to clone this repository and run setup. Alternatively, copy the entire prepared `data/ready/human-core-v1/` directory into another checkout; the loader needs no raw downloads. Relative paths in source metadata describe original acquisition provenance and can differ from setup's actual cache location. A chat AI needs actual filesystem access or the files attached; this document alone does not provide the dataset bytes.

For an export integrity check in any prepared checkout:

```bash
uv run --frozen python prepare.py --verify-only
```

For full verification against raw inputs in the original workspace:

```bash
uv sync --frozen
uv run python diverse.py verify
uv run python diverse_core.py verify
```

Those commands additionally need the broader release, pinned raw files and project configuration. Existing releases are preserved: build commands require a fresh output directory, and rebuilding is unnecessary for consuming these saved exports. See [README.md](README.md) for acquisition, reproduction and remaining provenance/licensing work.

## Suggested instruction for another AI

> Read DATA_HANDOFF.md in this repository. Run `uv run --frozen python prepare.py` to prepare the exact recommended core, then load training text from `data/ready/human-core-v1/train.jsonl.gz`. Preserve held-out partitions and source metadata. Keep translation controls separate, and describe the corpus as high-likelihood human-origin rather than certified human-only.
