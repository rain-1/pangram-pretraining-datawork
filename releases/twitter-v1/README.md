# Twitter detector controls v1

Prepare with `uv run --frozen python prepare_twitter.py`. This release selects controls for detector research and does not certify human-only text or extend the pretraining corpus.

`release.json` pins `source.json`, `selection.json` and `licenses.snapshot.json`. Sources are pinned to immutable upstream revisions with byte counts and SHA-256 hashes. The selection pins canonical JSON record hashes for all 105,312 audit records, the 64,466 deduplicated binary examples and 7,207 quarantined records, plus label/split counts. Hashing uses one compact, key-sorted UTF-8 JSON record plus newline per record, in record-ID order; text and all provenance fields participate. No tweet prose is stored in this recipe.

The authoritative importer is `twitterwork.py`; the portable preparation and verification entry point is `prepare_twitter.py`. It reads TweepFake's semicolon-delimited CSV with proper quoted-newline handling and Unmasking's JSON Lines, despite the `.json` filenames. It requires alternating human/generated pairs and identical reference sequences across model variants. It preserves released text, using NFKC/casefold/whitespace normalization only for identity grouping.

| Source | Label 0: provisional reference | Label 1: generated |
| --- | ---: | ---: |
| TweepFake | 12,781 | 7,945 |
| Unmasking | 4,194 | 39,546 |

TweepFake positives use account labels `gpt2` and `rnn`. Its 4,840 `others` bot records stay out of binary controls. Human labels remain provisional. Unmasking positives cover all nine released model variants; human references are deduplicated across variants. Across the collection, 2,367 records in normalized-text groups with both human and generation labels are quarantined. These are label conflicts, not a determination that either label is false.

Every account, prompt pair and normalized exact-text group stays in one split. Whole TweepFake account components are assigned deterministically, stratified by eligible-label composition, with roughly 15% of groups each for validation/test. Unmasking follows the highest held-out priority among connected original partitions (test, then validation, then train). Cross-source text matches merge components. The original upstream split is retained for audit. This does not recover missing reference authors or bot/imitated-author relationships, and approximate duplicates are not screened.

| Split | Reference | Generated | Total | Independent groups |
| --- | ---: | ---: | ---: | ---: |
| Train | 10,925 | 37,618 | 48,543 | 3,273 |
| Validation | 2,552 | 4,985 | 7,537 | 439 |
| Test | 3,498 | 4,888 | 8,386 | 448 |

Ready binary files are `data/ready/twitter-v1/{train,validation,test}.jsonl.gz`. Use `text` and `binary_target`; preserve metadata separately. Audit and quarantine exports overlap source records and are not extra training examples. `manifest.json` records output checksums; reruns verify frozen selection and splits, and mismatches fail without overwriting inputs. `--offline`, `--verify-only`, `--cache` and `--output` are supported.

The author sources are [TweepFake](https://github.com/tizfa/tweepfake_deepfake_text_detection), [Fagni et al.'s paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251415), [Unmasking's release](https://huggingface.co/datasets/redasers/Unmasking-the-Imposters) and [its COLING paper](https://aclanthology.org/2025.coling-main.607/). Dataset labels provide useful controls; no source establishes that every human-labelled tweet is independently proven human-written. Declared dataset licences do not independently clear all tweet-content rights. The licence snapshot records that distinction and applicable removal paths.

## Optional GrokSet

`source.json` also pins the optional [GrokSet](https://huggingface.co/datasets/bercev/GrokSet) dehydrated release. The default Twitter command downloads its small README, without the 1.05 GB metadata file. `uv run --frozen python prepare_grok.py metadata` downloads/indexes that file separately. The observed pin contains 160,802 conversations, 182,707 threads, 1,146,454 tweet occurrences and 1,053,541 unique IDs. The index identifies 252,029 English, non-media-only assistant replies. These are observed file counts; some differ from the authors' summarized counts. The publication/collection date is not the authorship date of every included post.

IDs and metadata have no usable tweet text. Generated annotations are discarded. The optional `import-text --input PATH` command accepts local JSONL with explicit `id`, `original_text` and optional `lang`. It checks the metadata role and English reply constraints, excludes other participants, deduplicates text and partitions connected conversations. The [handoff](../../DATA_HANDOFF.md#twitter-detector-controls) describes the schema, checks and limits. No tweet text has been hydrated in this workspace; the importer never requests platform access. Grok outputs have only positive labels, so they require separate suitable reference controls before evaluation. Cross-source combinations need overlap checks and regrouping.

GrokSet's metadata/annotations licence is declared CC BY-NC 4.0. Tweet content rights and platform access terms remain separate, and local text authenticity is not established by ID matching alone. Reruns verify the locally prepared metadata manifest and checksums; local hydrated selections have their own input-dependent manifests rather than a frozen tweet-text selection in Git.
