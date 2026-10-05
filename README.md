# Human-origin pretraining datawork

This project separates two jobs: selecting pretraining text using provenance, and collecting labelled machine translations to evaluate a filter. The translation benchmarks are kept out of the pretraining corpus inputs.

For an AI or developer consuming the exact saved release, start with [DATA_HANDOFF.md](DATA_HANDOFF.md): pinned file paths, a verified Python loader, split boundaries and the separate translation controls.

## Prepare the selected data after cloning

With [uv](https://docs.astral.sh/uv/) installed, run:

```bash
git clone https://github.com/rain-1/pangram-pretraining-datawork.git
cd pangram-pretraining-datawork
uv run --frozen python prepare.py
```

The command installs the locked Python dependencies, downloads pinned public inputs, reconstructs the **exact selected 1,109,431 words in 463 passages**, and verifies the text, metadata and existing splits. Ready files are in **`data/ready/human-core-v1/{train,validation,test}.jsonl.gz`**. Train contains 404 passages / 969,765 words; validation contains 23 / 51,977; test contains 36 / 87,689. Read each record's `text` field. This is the recommended diverse core, not the entire 45.5-million-word candidate pool.

Setup needs about 13 MB of book/WikiText downloads plus a streamed Hansard shard prefix (about 17 MB uncompressed); dependency downloads and HTTP read-ahead add overhead. It requires no paid corpus access or translation benchmark downloads. Raw files are cached under `data/raw/ready-core-v1/`. Rerunning verifies the completed export without downloading again. Use `--verify-only` for an explicit export check, `--offline` with a populated cache, or `--cache` / `--output` for other paths.

The repository contains code, licence records and the [frozen selection recipe](releases/human-core-v1/README.md), not corpus prose or raw archives. The recipe preserves source locations, passage hashes, membership, order and partition groups. Portable exports use relative provenance paths and deterministic gzip headers, so their metadata paths/archive checksums differ from the original local export while selected wording and splits are identical. Live Gutenberg URLs can change: a changed source fails its pin rather than silently producing different data. Internet access and continuing availability of the pinned upstream files are required for a first setup.

Human origin remains a provenance inference, and some licence questions remain unresolved; see the [handoff](DATA_HANDOFF.md#licences-and-portability). Only this core is prepared by the quickstart. The larger candidate pool and detector controls described below require their separate workflows. Earlier local `data/` artifacts are not included in a clone.

The quickstart was tested in a fresh checkout with no local data: every upstream pin matched, all selected text and partitions matched the original core, and all 61 project tests passed. Run the tests with `uv run --frozen python -m unittest discover -s tests`.

## Prepare the Google Translate controls

In the same checkout, run this separate command:

```bash
uv run --frozen python prepare_translations.py
```

It downloads the pinned official Par3 archive (285 MB), extracts the exact **1,055 Google Translate outputs and 2,272 human translations**, and verifies selected English text, foreign source paragraphs, process labels and work-level splits against the [frozen control selection](releases/google-translate-v1/README.md). Outputs are in **`data/ready/google-translate-v1/`**. Use `binary_examples.jsonl`, reading `text` as the English input and `binary_target` as the label: **1 = machine translation, 0 = human translation**. Respect each record's `split`; train/validation/test contain 3,058 / 208 / 61 examples. Every paragraph is 400–3,000 words, and no unrelated release rows are joined.

Rerunning verifies the export without downloading or extracting again. The archive is cached at `data/raw/par3/par3.pkl`; `--offline`, `--verify-only`, `--cache` and `--output` work like the core setup. The repository contains control IDs, hashes, source configuration and licence records, not translated prose. Internet access and continuing availability of the official download are required for a first setup. The archive is parsed with callable/global pickle construction disabled; its pinned checksum is verified before parsing.

These are detector research controls, **not pretraining inputs**. Human translations are eligible human negatives. Individual literary translation rights, Google engine versions and generation dates remain unverified. The existing experimental detector is not prepared by this command and should not drive corpus removal: it falsely flags 71.3% of these human translations.

Translation setup was tested in a fresh checkout with no archive: the official download matched its pin, every control and partition matched the original selection, and all 66 project tests passed.

**The current v2 corpus contains 52,186 contiguous passages of 400–3,000 whitespace words, totalling 45,507,097 words.** It uses the downloaded news and Hansard, with source-specific footer cleanup and verified near-copy removal. The v1 artifacts remain preserved. Separate Par3 controls now supply 1,055 long Google Translate outputs and 2,272 human translations. The frozen experimental detector fails badly on those human controls and is excluded from corpus removal. These are pretraining candidates; no human-only guarantee or corpus contamination estimate has been measured. NEWSROOM files remain pending, and no paid Gigaword files are needed for this version.

## Diverse human-origin core

The recommended diverse documentary subset is `data/corpus-human-diverse-core-v1/`: **1,109,431 words in 463 passages**, each within 400–3,000 words. It retains historical human-authored works, the documented 2016 WikiText reference collection, and selected named parliamentary prose. The previous parliament-only core remains available separately.

| Writing family | Passages | Words | Share |
| --- | ---: | ---: | ---: |
| Reference / encyclopedia | 215 | 444,959 | 40.1% |
| Fiction | 99 | 293,861 | 26.5% |
| Nonfiction: memoir, essays, writing instruction | 76 | 222,519 | 20.1% |
| Parliamentary written Q&A | 73 | 148,092 | 13.3% |

Use `train.jsonl.gz`, `validation.jsonl.gz` and `test.jsonl.gz`, reading each record's `text` field. Human origin is inferred from documentary provenance, not certified for each passage. Historical works are original English by identified authors; Wikipedia provides more recent reference language, but individual authorship/revision IDs remain unavailable. The genre mix supplies narrative, personal and explanatory writing alongside formal institutional prose. It does not establish comprehensive topic coverage or demographic representativeness.

The broader `data/corpus-human-diverse-v1/` contains **1,479,980 words in 931 passages**. It adds 370,549 words of named-author journalism from seven publishers. News requires a named metadata author also present near the article beginning, or an explicit reporting/writing credit, publication through 2018, acceptable known-update dates, and a whole-parent scan for generation/translation/extraction mentions. That is stronger selection than the general v2 pool, but bylines and publication dates do not prove an unchanged human-only captured version. This news remains outside the recommended documentary core, with its confidence basis and rights uncertainty visible.

The broader mixture targets reference 30%, fiction 20%, nonfiction 15%, journalism 25% and parliament 10%. Its total size is set by the limiting eligible genre capacity, not a requested word cap. Selection uses whole passages without repetition, bounds each news publisher to approximately 8% of the target, individual book authors to 12%, and individual non-book documents to 2%. Underfilling by whole passages produces small deviations from target shares. The documentary core is an exact four-genre view of that mixture, so its shares differ as shown above.

New inputs are pinned in `config/diverse_downloads.json`: the official Salesforce WikiText-2 raw distribution at revision `b08601e04326c79dfdd32d625aee71d232d685c3` (upstream LFS hashes), and eleven Gutenberg plaintext books (observed first-acquisition hashes). Books include works by Austen, Shelley, Carroll, Wells, Brontë, Conan Doyle, Stevenson, Douglass, Thoreau, Strunk and Mill. Import uses only content inside ebook body markers: catalog summaries and reading guides are excluded, including Gutenberg's explicitly generated webpage summaries. Ebook acquisition/update dates are distinguished from historical publication years. Raw downloads remain under `data/raw/diverse-human/`.

WikiText derives from Good/Featured Wikipedia articles and was introduced in 2016; the pinned Parquet conversion is from 2024. Adjacent release rows are joined only within a single identified article. Article boundaries require blank-separated first-level headings. This keeps inline equations and table-key equalities in their parent article; the importer recovers 600 train, 60 validation and 60 test articles. Formatting restoration mechanically removes `@-@` / `@,@` / `@.@` markers, repairs punctuation spacing and common contractions. It generates no wording; remaining tokenizer formatting is a potential detector shortcut. Both the raw article hash and the processed parent hash are retained. Source URLs are inferred from mechanically restored titles, not verified revision links. [WikiText source card](https://huggingface.co/datasets/Salesforce/wikitext), [2016 paper](https://arxiv.org/abs/1609.07843).

All output records retain candidate labels and null binary targets. `human_origin_basis` distinguishes historical works, the frozen reference release, parliamentary provenance and named web bylines. A passage-level similarity pass verifies retrieved seven-word-shingle pairs and groups connected parents before partitioning, giving existing test/validation material priority over train if groups connect. The current build found no cross-source pairs at its 0.80 threshold; approximate retrieval is not a guarantee that all similar text was found. Whole books/author groups and article parents stay together. Do not concatenate these overlapping exports with each other, the old core or general v2.

```bash
uv run python fetch_diverse.py
uv run python diverse.py build
uv run python diverse.py verify
uv run python diverse_core.py build
uv run python diverse_core.py verify
```

Build commands require fresh output directories; use `--output` when reproducing an existing release. Acquisition refuses changed raw pins. Verification checks raw/export hashes, deterministic allocation, reconstructed source bodies and contiguous slices, unchanged core membership, unique text and parent partitions. Frozen license snapshots allow verification after the live inventory gains new entries. The completed releases passed integrity checks, and all 53 project tests passed. `diversity.json`, manifests, parent-document exports, source evidence and processing snapshots preserve the source/author/genre inventories and lineage. Parent-document exports overlap the span text and should not be added as separate training examples.

Licenses and all derived paths are tracked in `config/licenses.json`. Gutenberg's USA public-domain declarations are distinguished from its ebook/trademark terms and other jurisdictions. The WikiText card itself conflicts between CC BY-SA 3.0/GFDL in its header and CC BY-SA 4.0 in its body; reconciliation remains open. News license/version uncertainties remain unchanged. No translation-benchmark positives are included in either mixture. [Gutenberg terms](https://www.gutenberg.org/policy/license.html).

## Preserved parliament-only core

`data/corpus-human-core-v1/` is a separate, substantially stricter **subset of v2**: **104 unchanged English passages, 221,582 whitespace words, from 83 parent records dated 1999–2010**. Use its `train.jsonl.gz`, `validation.jsonl.gz` and `test.jsonl.gz`, reading each record's `text` field. Do not concatenate this subset with v2: its text already exists there.

The core retains named House of Lords written questions and ministerial answers. Parliament describes these as questions from members to ministers, and contemporary official guidance describes civil servants drafting answers for ministers. This supports a high likelihood of human-origin prose, particularly with the 2010 publication cutoff, while leaving undisclosed assistance and later text-version uncertainty unresolved. These are **written Q&A records, not spoken-debate transcripts**. [Written-question process](https://www.parliament.uk/about/how/business/written-answers/), [historical drafting guidance](https://publications.parliament.uk/pa/cm200405/cmselect/cmpubadm/449/44905.htm).

`config/human_core.json` requires agreement between the date-bearing record ID, publication date and metadata year; English provenance; the parliamentary license; at least two named participants; an explicit question followed by a named answer; and passages within 400–3,000 words. A whole-parent scan quarantines generation/translation mentions and recognisable extraction annotations. Prose screens exclude passages containing substantial numeric/tabular material. These are conservative selection rules, not machine-authorship labels: many excluded passages are human-written. All web news and translation benchmark records stay outside this strict core. No learned detector or text rewriting is used.

| Partition | Passages | Words |
| --- | ---: | ---: |
| Train | 102 | 218,027 |
| Validation | 1 | 568 |
| Test | 1 | 2,987 |

The small held-out counts result from preserving the v2 parent/similarity partitions; these files alone are insufficient for a reliable detector evaluation. The genre is narrowly institutional and should not be treated as representative of general English. Original `human_candidate` labels, null binary targets, review status, flags and captured-version uncertainty remain intact. `human_core_selection` records each passage's source rationale, participants and prose-screen metrics; it does not certify individual authorship or estimate purity.

```bash
uv run python strictsubset.py build
uv run python strictsubset.py verify
```

Build requires a fresh output directory; use `--output` to reproduce the selection elsewhere. Verification checks source/output/evidence hashes, every passage's exact equality with v2 and its parent slice, policy compliance, counts, unique text and unchanged partitions. All 45 project tests passed. The manifest, exclusion log and verification report are in the output directory. Original corpus files are preserved.

An assistant spot check of one selected 1999 record found matching prose in the official historical archive; `data/human-core-archive-check.json` records the comparison. This corroborates one record, not all 83 parents or their captured revision dates. Source workflow and license evidence is saved under `data/raw/human-core-evidence/` as tool-rendered retrieval snapshots; direct HTTP retrieval returned 403, so these are not original HTML snapshots. A provisional first pass was superseded after spotting extraction notes; its processing code, policy and summary remain recorded separately. [Official 1999 answer archive](https://publications.parliament.uk/pa/ld199900/ldhansrd/vo991118/text/91118w01.htm).

The output includes `ATTRIBUTION.txt` and an updated license inventory snapshot. The Open Parliament Licence requires attribution and has exceptions for personal data and third-party rights; the core is not an independent rights certification. [Official license](https://www.parliament.uk/site-information/copyright-parliament/open-parliament-licence/).

## Current v2 workflow

After the v1 acquisition and curation below, reproduce v2 into fresh output directories:

```bash
uv run python quality.py
uv run python verify_quality.py data/curated-v2
uv run python wordcorpus.py --curated data/curated-v2 --output data/corpus-words-v2
uv run python verify_wordcorpus.py data/corpus-words-v2
```

The recommended passage files are `data/corpus-words-v2/{train,validation,test}.jsonl.gz`. No overall word cap was supplied, so every eligible passage is retained once. Do not combine them with the v1 partitions or the full-document exports: those contain overlapping text and v2 reassigns some parent groups.

| Partition | Passages | Whitespace words |
| --- | ---: | ---: |
| Train | 51,093 | 44,534,158 |
| Validation | 544 | 493,335 |
| Test | 549 | 479,604 |
| Total | 52,186 | 45,507,097 |

`quality.py` reads the frozen v1 full documents and `config/quality.json`. It trims recognised footer tails from **5,518 documents**, removing **287,422 words**. It only truncates an exact suffix when every remaining nonempty line is recognised footer or repeated related-story material; article prose following an inline newsletter/survey widget is preserved. The retained body is an exact prefix of the original, and each removal is logged with the removed text and source character range. After cleanup, 485 documents fall below 400 words and one article with a possible Google-translation disclosure is quarantined for review. That disclosure can concern excerpts or images, so it is not a confirmed machine-text label.

For near copies, 128-permutation MinHash retrieves candidate pairs using seven-word shingles; actual string-shingle Jaccard similarity verifies each decision. Pairs at 0.80 or above share a partition. A document is removed only when a directly retained copy has at least 0.92 similarity and a length ratio of at least 0.85. Connected similarity groups control partitioning, not blanket deletion: a transitive chain alone cannot remove its endpoints. The pass confirms **1,044 similar pairs**, removes **369 near copies**, and moves **34 surviving documents** between partitions. Fifty-five confirmed pairs crossed v1 partitions. Approximate retrieval can miss pairs, so this does not certify that all near duplicates have been found. See the [MinHash LSH documentation](https://ekzhu.com/datasketch/lsh.html).

`data/curated-v2/` retains 51,561 full documents and their separate contiguous spans. `data/quality-v2/` contains similarity scores, direct-copy decisions, tail changes, per-publisher cleanup counts and the disk-backed document inventory. These audit exports also contain dataset text and are tracked in the license inventory. The initial provisional quality pass is superseded; its code/config and summary are preserved as lineage, not recommended corpus files.

`verify_quality.py` checks hashes, unchanged source metadata, exact original body/span slices and word bounds, and independently recomputes all 1,044 confirmed pairs and 369 direct deletions. `verify_wordcorpus.py` checks the assembly, parent/similarity-group separation, unique normalised passage text, counts and license annotations. Both completed successfully. The 235-record provenance review queue remains unreviewed; date selection, deduplication and cleanup do not establish authorship.

## Source-policy and license observations

The current `config/licenses.json` and v2 assembly license snapshot record publisher observations separately from the original dataset labels. Common Pile declares CC BY 4.0 for Global Voices and Alt News, while their official pages currently display **CC BY 3.0**. These discrepancies are marked `license_version_mismatch_requires_reconciliation`; the original `license` and `metadata.license` values are preserved, with a separate `publisher_license_observation` on affected v2 passages. Publisher declarations also do not automatically cover third-party material. Saved policy pages, URLs, acquisition dates and hashes are in `data/raw/news/source-policies/`. [Global Voices attribution policy](https://globalvoices.org/about/global-voices-attribution-policy/), [Alt News about page](https://www.altnews.in/about/).

Global Voices describes a volunteer writing/translation process, and its current AI policy rejects automatically generated writing and translations. The policy was updated in September 2026; it does **not** retrospectively certify the authorship of our captured articles. Human translations remain eligible. Individual article/version evidence still needs review. [Global Voices about page](https://globalvoices.org/about/), [AI policy](https://globalvoices.org/about/global-voices-policy-on-ai/).

CC-News article rights remain unresolved. No individual English translation rights have been established for Par3. Keep these pools distinguishable, and use the inventory's raw/derived paths when deleting a source or rebuilding affected exports. The assembly's license snapshot includes the new observations; the earlier immutable curation snapshots preserve their original inventory state.

## Long Google Translate and human controls

The official [Par3 release](https://github.com/katherinethai/par3) supplies English Google Translate outputs aligned with human literary translations. `config/par3.json` pins repository revision `04ca269d2e46098eaad0cdca34c10b8580f1c047`, the official public download route and the observed 285,335,827-byte archive checksum. The archive has no published upstream cryptographic checksum; its local pin records the first acquisition. The repository's MIT license concerns software/documentation and does not establish rights to every English translation.

```bash
uv run python par3work.py fetch
uv run python par3work.py build --output data/detector-par3-v2
uv run python par3work.py score --output data/detector-par3-v2 --model data/detector-v1
uv run python verify_par3.py data/detector-par3-v2
```

The importer rejects pickle globals/callable construction and reads only the release's built-in data containers. The actual archive has 113 book/volume entries and 122,819 aligned paragraph positions. Its paragraph order is shuffled and human translations are only partially retained, so **no release rows are concatenated**. A native Google paragraph is retained only when both it and at least one aligned human variant independently meet the 400–3,000-word bounds. Verification compares every exported source/target paragraph against its original archive position.

`data/detector-par3-v2/` contains **3,327 deduplicated long controls: 1,055 Google Translate outputs and 2,272 human translations**, spanning 77 book/volume entries, 69 connected work groups and 12 source languages. Identified volumes/trilogies and exact matches share a partition. Fixed hash ranking assigns 55 work groups to train and seven each to validation and test. Paragraph availability is uneven: those splits contain 3,058, 208 and 61 examples respectively. The small test set supports only a limited future comparison; authors and source languages are not held out. The preliminary `detector-par3-v1` partitioning is superseded and should not be mixed with v2.

The frozen v1 baseline was evaluated on all 3,327 Par3 controls without retraining or threshold tuning. Even Par3's train partition is external to that model. It falsely flags **1,620/2,272 human translations (71.3%)**, while recognising **884/1,055 Google outputs (83.8%)**. Resampling whole work groups gives 95% bootstrap intervals of **63.2–78.1%** for human false positives and **76.7–89.6%** for machine recall. These results concern this literary benchmark, not contamination or detector performance on the news corpus. They demonstrate why the baseline cannot drive removal. Scores, evaluation, uncertainty method and model/input/output checksums are saved separately from extraction in `external_v1_*` files. Generation dates and the exact Google engine version remain unknown.

## Downloaded full-length news sources

The public CC-News and Common Pile News releases are fully downloaded under `data/raw/news/`, with immutable repository revisions, upstream file SHA-256 checksums and original metadata. The 22 archives total 1,277,961,765 bytes. Repeat the acquisition and inspection with:

```bash
uv sync --frozen
uv run python newswork.py fetch
uv run python newswork.py inspect
```

`config/news_downloads.json` pins all file URLs, sizes and upstream checksums. `data/news/acquisition.json` records verification status, and `data/news/inspection.json` inventories all rows. Inspection uses bounded Parquet batches/gzip streaming. The raw archives remain unchanged. The curation pass below selects candidates without rewriting their prose; human authorship remains unconfirmed.

| Download | Local directory | Rows | Rows with at least 400 whitespace words |
| --- | --- | ---: | ---: |
| CC-News English | `data/raw/news/cc_news/` | 708,241 | 257,721 |
| Common Pile News | `data/raw/news/common_pile_news/` | 172,308 | 52,992 |

These length counts are before date, provenance, licensing or duplication filtering and are not model token counts. The CC-News card describes 2017–2019 coverage; the actual parseable publication dates in this pinned release comprise 359,647 rows in 2017, 270,994 in 2018 and two in 2019. Another 77,598 rows have missing/invalid date values, so don't assume every row satisfies a date cutoff. Common Pile News has heterogeneous publication/update strings and recent articles that require source-specific handling.

`config/licenses.json` tracks every acquired dataset plus planned licensed sources, including unknown rights, original raw paths and shared derived exports. Original Common Pile News per-document `metadata.license` fields remain intact. Declared license labels have not been independently verified for each article. CC-News licensing remains unspecified. Saved source cards and access agreements, with response hashes, are under `data/raw/news/documentation/`.

The user confirmed non-commercial research use for NEWSROOM and reports completing its access form personally. Its article archive is **not yet acquired**; a download link or local files are still needed. The NIST Reuters collections are **excluded at the user's request** because of the organizational paperwork; no RCV1/RCV2/TRC2 archive was acquired. Separately, some downloaded CC-News web articles originate at reuters.com and retain unknown article rights; they are not those licensed NIST releases. `data/news/access_status.json` records the current state separately from the frozen acquisition configuration, and `config/licenses.json` tracks the licensing evidence. Prepared instructions remain in `requests/newsroom-access.txt` and `requests/reuters-access.txt`. Once NEWSROOM access files are supplied, pin/checksum them and add its source-specific adapter. No external access request has been submitted by the agent.

## Preserved v1 curation and word-counted corpus

The v1 stage supplies the input for v2. Reproduce it offline after acquisition:

```bash
uv run python curate.py --output data/curated-v1
uv run python verify_curated.py data/curated-v1
uv run python wordcorpus.py --output data/corpus-words-v1
uv run python detector.py --output data/detector-v1
```

The completed first pass scanned **881,049 rows**, retained **52,416 documents**, and exported **53,044 large passages containing 46,227,307 whitespace words**. It removed **3,647 duplicate articles or alternate captures**. Full documents contain 46,251,462 words; the difference is short final tails omitted from passage exports. All exported passage hashes, source slices, provenance, lengths and splits passed verification, with no exact duplicate passage text found.

| Source | Retained documents | Large passages | Passage words |
| --- | ---: | ---: | ---: |
| Common Pile News | 37,001 | 37,091 | 32,969,346 |
| Selected CC-News publishers | 15,008 | 15,050 | 11,043,347 |
| Hansard prefix | 407 | 903 | 2,214,614 |

This is an inventory selection without source quotas: Global Voices supplies about 45% of passage words. CC-News contributes roughly 11 million words with unresolved article licensing, and the source files keep that pool distinguishable. The 235 publisher-stratified review records remain unreviewed; none of these candidate documents has been promoted to a certified-human label.

Each command requires a fresh output directory, so use a different name when rerunning. `config/curation.json` specifies a **400-word minimum**, **3,000-word maximum per passage**, English checks on the beginning/middle/end, a publication cutoff of **2021-12-31**, and the selected CC-News publishers. This cutoff is a conservative curation choice, not proof that older text is human-written. The language detector runs locally with bundled models; its confidence is a language score, not an authorship score.

Common Pile News dates use source-specific publication strings and dated URLs, preserve explicit update dates, and reject ambiguous/missing/conflicting dates or known updates after the cutoff. The actual URL host must match its source before the source license label is carried forward. CC-News uses its extracted publication field and a selected publisher list; the capture/version date and article licensing remain unknown. Human translations stay eligible. Explicit automatic-translation notices are quarantined for review, including the possibility of a quoted notice or disclosure about another article.

`data/curated-v1/` keeps full documents and separate large-span files for each dataset. Do **not** concatenate the full-document and span exports: they contain overlapping text. Spans preserve exact character slices and paragraph/sentence endings where possible. Short final tails stay in the full documents but are omitted from the passage export. Documents are grouped transitively by normalized full text or canonical article URL, then a declared-license copy and the longest eligible copy are preferred. Parent groups receive deterministic approximately 98/1/1 partitions. The v1 stage does not handle near copies; v2 adds that pass.

The curation manifest records every input checksum, policy, dependency versions, source counts and output hashes. A final-tail boundary correction rebuilt spans from unchanged full documents; `initial.manifest.json` and the `span_rebuild` lineage preserve the first-pass hashes, and the processing scripts are saved under `data/raw/news/documentation/`. Fresh runs of the corrected `curate.py` produce the corrected span behavior directly. `inventory.sqlite` retains eligible records before deduplication, so it also contains dataset text. Rejections and duplicate/article-variant decisions are logged separately. `review_queue.jsonl` holds eight deterministic sample articles per retained publisher/dataset where available, with empty fields for human-origin evidence, version evidence and translation workflow. It is not a completed human review or contamination audit. `verify_curated.py` verifies artifacts, source text/provenance, exact passage slices, word bounds and split separation.

`data/corpus-words-v1/` assembles passages into `train.jsonl.gz`, `validation.jsonl.gz` and `test.jsonl.gz`, preserves their source/license metadata and parent partitions, and deduplicates normalized passage text. No overall cap was supplied, so the first assembly retains all qualifying spans: train has 51,935 passages / 45,239,850 words; validation has 558 / 501,701; test has 551 / 485,756. The assembly also passed output-checksum, unique-text, word-bound and parent-group separation checks; `verification.json` records those results. To use only the declared-license source families or set a later word cap:

```bash
uv run python wordcorpus.py \
  --curated data/curated-v2 \
  --sources common_pile_news hansard \
  --word-budget 10000000 \
  --output data/corpus-words-subset
```

The cap above is illustrative. It covers all three splits, whole passages can underfill it, and no repetition is used to reach it. “Declared license” does not imply independent rights verification. `config/licenses.json` tracks raw files, source evidence and derived exports, including SQLite inventories and shared models; deleting a source requires rebuilding affected exports/models.

## Experimental translation baseline

`detector.py` combines the existing translation segments with 400-word-or-longer aligned spans, **regroups all source documents before splitting**, excludes conflicting workflow labels and deduplicates target text. It fits a character 3–5 gram TF-IDF/logistic baseline using English prose alone, fitting vocabulary and classifier only on train. Candidate news and Hansard are excluded from the human controls. A threshold is chosen on validation to allow at most 1% observed human false positives, then frozen before test evaluation.

The original v1 baseline release contains **21,909 binary examples**, including **660 large passages**. Its **600 named Google Translate examples are all short**; the separate Par3 release above now adds long named Google controls. `data/detector-v1/` contains grouped examples, scores, model, feature audit, source/license snapshot and evaluation report. Existing aligned spans were built using a demo tokenizer; selection for this experiment uses word counts and does not join unrelated segments.

The reference baseline achieved **12.6% machine/assisted recall** and **2.54% human false positives** on 2,726 held-out examples. For large passages it caught **19/53** machine/assisted examples and falsely flagged **3/20** human translations. Its observed validation target did not transfer to test; it is **not used for corpus removal**. The large test spans cover only six independent document groups; shared generators/domains and absent original-English controls prevent a deployment claim. Metadata and labels are excluded from model features, but text styles/topics can still become shortcuts; inspect `feature_audit.json`. Wilson intervals in the reports assume independent rows; a final audit needs document-cluster uncertainty.

## Run the original bounded pilot

```bash
uv sync --frozen
uv run python datawork.py fetch
uv run python datawork.py build
uv run python datawork.py verify
uv run python -m unittest discover -s tests -v
```

The fetch command downloads/validates frozen translation inputs and the first 500 complete rows of one pinned Hansard shard. It does not download the full Pile, Common Pile, or licensed Gigaword. Existing files must match recorded checksums. Raw data and generated outputs are excluded from Git.

| Prepared source | Records | Use |
| --- | ---: | --- |
| Hansard through 2021 | 439 | Human-origin candidates; 2,250,599 whitespace-separated words |
| WMT2021 Chinese → English | 9,681 | 1,294 human references and 8,387 system outputs |
| Collaborative translation, Chinese → English | 21,736 | Human, machine and assisted workflows |
| Spanish idioms → English | 1,800 | 600 outputs each from Google Translate, DeepL and GPT-4o |

The full pilot contains 33,656 records. These are local extraction counts, not the sizes of the upstream releases. Counts include translation alternatives and are not estimates of distinct pretraining text volume.

For larger passages, the prepared `translation_spans.jsonl` contains 680 aligned passages and `binary_spans.jsonl` contains 662 deduplicated examples between 512 and 4,096 GPT-2 tokens. Regenerate with the actual training tokenizer:

```bash
uv run python datawork.py spans \
  --tokenizer-json /path/to/training/tokenizer.json \
  --min-tokens 512 --max-tokens 4096
```

Spans join only contiguous segments within one source document, using the same segment boundaries for all translation variants. Missing segments break a span; unrelated documents are never concatenated. Short passages are excluded. Use the sentence or span export as a separate experiment; do not combine their partitions without regrouping. `spans.manifest.json` records its tokenizer/input/output hashes. Long-span availability is much smaller than the sentence-level pilot.

`data/pilot/` contains:

- `human_candidates.jsonl`: Hansard text with publication-date checks and retained metadata.
- `translation_examples.jsonl`: all retained translation variants with process labels.
- `binary_examples.jsonl`: 21,249 unique normalized texts for detector work: 4,732 documented human translations and 16,517 machine/assisted examples. Train/validation/test counts are 16,284/2,312/2,653.
- `records.jsonl`: the complete combined pilot with IDs, hashes, provenance, labels and partition assignments.
- `rejections.jsonl`: 61 Hansard rows dated after 2021 and 46 WMT groups with conflicting underlying annotation text.
- `review_queue.jsonl`: unfilled review fields for Hansard candidates. No human review is implied.
- `summary.json` and `manifest.json`: counts, limitations, input/output hashes, frozen URLs, revisions and processing-code hash.

## Optional legacy tokenizer-based selection

`corpus.py` streams local Gigaword, raw Common Pile Hansard, or merged original Pile JSONL/JSONL.gz inputs. It counts tokens with a supplied Hugging Face `tokenizer.json`, records its hash, disables truncation/padding, and optionally counts one EOS per document. It writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, a rejection log, a manifest, and a disk-backed SQLite exact-deduplication index.

`--token-budget` means the **total selected tokens across all three splits**. Train/validation/test use a deterministic approximately 98%/1%/1% hash assignment. If the budget refers only to training tokens, account for the held-out allocation separately before setting the total. Documents are retained whole, so the total can undershoot the requested cap. The manifest reports the actual token counts and unfilled amount. Insufficient source capacity is reported, never filled by repeating documents or introducing synthetic data.

Use the actual training tokenizer. The frozen GPT-2 tokenizer in `data/raw/` is only for the local smoke test.

Start without paid data, using the already downloaded Hansard prefix for a reproducible smoke test:

```bash
uv run python corpus.py \
  --input hansard=data/raw/hansard_prefix.jsonl \
  --tokenizer-json data/raw/gpt2-tokenizer.json \
  --token-budget 100000 --eos-token-id 50256 \
  --min-document-tokens 512 --cutoff 2021-12-31 \
  --evidence provenance \
  --output data/corpus-free-pilot
```

This uses the demonstration tokenizer and a small illustrative budget. For production, replace both, acquire more eligible sources, and set source quotas. Hansard provides long parliamentary documents; broaden the mix with reviewed openly licensed scholarly, legal and historical reference sources. Free access does not establish human authorship. These inputs retain candidate labels and need provenance review.

Optional example for licensed Gigaword files, only if access is already available (replace paths and the illustrative budget):

```bash
uv run python corpus.py \
  --input gigaword=/path/to/LDC2011T07/data \
  --tokenizer-json /path/to/training/tokenizer.json \
  --token-budget 100000000 \
  --evidence documented \
  --output data/corpus-gigaword
```

For a mix, provide an explicit quota for each adapter, summing to the total:

```bash
uv run python corpus.py \
  --input gigaword=/path/to/LDC2011T07/data \
  --input hansard=/path/to/raw/hansard \
  --input pile=/path/to/original-pile \
  --quota gigaword=60000000 --quota hansard=10000000 --quota pile=30000000 \
  --tokenizer-json /path/to/training/tokenizer.json \
  --token-budget 100000000 --cutoff 2021-12-31 \
  --pile-snapshot-date 2020-12-31 \
  --evidence provenance \
  --output data/corpus-mix
```

The budgets and snapshot date in that command are examples. Verify the actual Pile archive's capture dates before supplying them. The script cannot infer an archive's vintage from its name. A source-level snapshot bound does not prove human authorship or exclude older MT.

The Gigaword adapter keeps `type="story"` bodies, parses dates from document IDs, decodes SGML entities and preserves paragraph order. Story labels are heuristic, so tables and boilerplate still require sampling. The Hansard adapter restricts the initial selection to explicitly English Westminster Commons/Lords sources, checks date-bearing IDs against `metadata.year`, and never substitutes ingestion date for publication date. Both publication and captured-version uncertainty remain visible.

The Pile adapter initially allows FreeLaw, NIH ExPorter, PubMed Central, PubMed Abstracts, ArXiv, PhilPapers, StackExchange and Wikipedia (en). This is a conservative source-selection policy, **not a set of human-authorship labels**. All other components are excluded, including DM Mathematics, broad web sources, subtitle/transcription sources, and sources requiring separate generated-file or bot-output handling. Pile component rights remain source-specific.

`documented` mode currently accepts only Gigaword's documented manual-origin source selection. `provenance` mode also permits Hansard/Pile candidates and labels them as candidates. Neither mode is a measured human-only guarantee. The builder applies source/date rules and exact deduplication. An experimental MT baseline is now measured separately; it is not used to remove corpus text.

The builder consumes files in sorted path order and documents in input order. This is a reproducible first selection, not random sampling. For production, shuffle/stratify eligible document inventories before budget selection, inspect source/year coverage, and add near-duplicate/version deduplication. Hash partitioning alone does not establish publisher, author or topic independence.

Documents must contain at least 512 tokenizer tokens by default; change `--min-document-tokens` for the chosen model/context. The large-document smoke test in `data/corpus-large-demo/` selected 21 whole Hansard documents containing 99,841 GPT-2 tokens, including one EOS per document, against a 100,000-token cap. It undershot by 159 tokens. This tiny selection happened to place every document in train and supplies no held-out evaluation set. It is a pipeline check, not a production corpus or measured authorship audit.

## Sources to scale

| Source | Recommended role and selection rule | Evidence/access |
| --- | --- | --- |
| English Gigaword Fifth Edition (optional; outside the current no-fee plan) | Possible news addition if licensed access becomes available. Start with story bodies; deduplicate updates and repeated transmissions. Count with the actual tokenizer. | The official documentation describes manually generated source text and reports about 3.67B whitespace tokens in stories. Licensed access is required. [LDC documentation](https://catalog.ldc.upenn.edu/docs/LDC2011T07/0readme.txt), [catalog](https://catalog.ldc.upenn.edu/LDC2011T07) |
| `walzen/gigaword` (reviewed; not imported) | Short sentence/headline benchmark, unsuitable as a replacement for full news stories in the large-span corpus. | The [dataset card](https://huggingface.co/datasets/walzen/gigaword) traces approximately four million pairs to the Rush summarization preprocessing. The [original paper, section 7.1](https://aclanthology.org/D15-1044.pdf) uses first sentences, averaging 31.3 words, lowercases text, masks digits and replaces rare words with `UNK`. The uploader declares Apache-2.0; underlying news redistribution/training permission has not been independently verified. |
| Common Pile UK Hansard | Parliamentary candidates. Restrict source channels and publication years; independently check captured text versions when required. | Source-separated records and year/language metadata. [Dataset card](https://huggingface.co/datasets/common-pile/uk_hansard) |
| Common Pile News (downloaded; candidate adapter implemented) | No-fee, openly licensed news candidates; use individual publisher files and review actual URL hosts, article/version dates and license provenance. | 172,308 extracted documents; per-document author, URL and license metadata. Includes 2023–2024 articles, so the release as a whole is ineligible for a pre-2022 rule. The `created` value can contain publication and update dates or be null. [Dataset card](https://huggingface.co/datasets/common-pile/news) |
| CC-News English subset (downloaded; candidate adapter implemented) | Convenient no-fee download of article bodies for publisher-by-publisher screening; keep candidates separate from documented human controls. | 708,241 English articles reported as published in 2017–2019; about 1.12 GB compressed. Its publication-date field is not a capture timestamp, and the dataset card leaves licensing unknown. Actual missing-date counts are recorded above. [Dataset card/files](https://huggingface.co/datasets/vblagoje/cc_news) |
| Cornell NEWSROOM (user completed access form; archive pending) | Stronger documented newsroom-origin candidate; use full `text` bodies and archived dates. Available free of charge for non-commercial research/education after agreement. | 1.3 million article/summary pairs from 38 publications, collected through 2017; project describes articles and summaries as written by authors/editors. [Project](https://lil.nlp.cornell.edu/newsroom/index.html), [download request](https://lil.nlp.cornell.edu/newsroom/download/index.html), [usage agreement](https://lil.nlp.cornell.edu/newsroom/terms/index.html) |
| Reuters TRC2 / RCV1 (excluded by user) | Reviewed alternative; not part of the current acquisition plan because of organizational paperwork. | TRC2: 1,800,370 stories from 2008–2009; RCV1: about 810,000 stories from 1996–1997. [NIST access](https://trec.nist.gov/data/reuters/reuters.html), [agreement](https://trec.nist.gov/data/reuters/org_appl_reuters_v4.html) |
| Common Pile Caselaw | Legal candidates. Join case identifiers to judgment metadata and keep identifiable older decisions. | The preview's `created` field can be 2024 for a 1973 opinion. A universal created-date cutoff fails. [Dataset card](https://huggingface.co/datasets/common-pile/caselaw_access_project) |
| Common Pile peS2o | Scholarly candidates. Keep paper IDs, PDF hashes, license metadata and archived/versioned source files. | Publication dates alone do not establish the PDF version retained. [Dataset card](https://huggingface.co/datasets/common-pile/peS2o) |
| Selected original Pile components | Technical breadth after provenance and rights checks. Preserve the original downloaded snapshot. | Merged records expose `meta.pile_set_name`, with less detailed provenance than upstream collections. [Dataset schema](https://huggingface.co/datasets/EleutherAI/pile/raw/main/README.md), [Pile paper](https://arxiv.org/abs/2101.00027) |
| Historical Stack Exchange / Wikipedia | Additional explanation and discussion. Prefer genuinely historical dumps or retained revision dates. | Aggregated Stack Exchange dates can describe the question while later answers/comments are included. Wikimedia's retained revision date describes its retained text. [Stack Exchange conversion](https://github.com/r-three/common-pile/blob/main/sources/stackexchange/preprocess.py), [wiki conversion](https://github.com/r-three/common-pile/blob/main/sources/wiki/to_dolma.py) |

Gigaword, Hansard and selected Pile components have runnable selection adapters. CC-News and Common Pile News have source-specific candidate adapters in `curate.py`. NEWSROOM, Reuters, Caselaw and peS2o are not yet acquired and need source-specific adapters before inclusion. No paid licensed release has been purchased or accessed through an unauthorised mirror.

For the immediate no-fee news plan, use reviewed Common Pile News sources when clear source licensing is required, or investigate historical CC-News publisher subsets when web-derived text is suitable. NEWSROOM supplies particularly useful newsroom-origin documentation if its non-commercial terms match the project. The first three rows inspected from Common Pile's Milwaukee shard point to external publishers and contain only 18–28 whitespace words despite source/CC-BY labels; do not propagate a site-level license automatically to outbound links or treat every extracted record as a complete story. Length, actual-host and rights checks are necessary.

## Machine-translation data and labels

The pilot uses these primary releases:

- [Google's WMT MQM release](https://github.com/google/wmt-mqm-human-evaluation) and [WMT2021 task documentation](https://www.statmt.org/wmt21/metrics-task.html): human references and system translations of shared source text. Multiple error/rater annotations are collapsed to a single underlying output. Only `<v>` annotation wrappers are removed; the erroneous prose is retained. Conflicting text versions are excluded for review. Anonymous online/development system names retain their released names and are not guessed to be Google Translate.
- [Collaborative translation release](https://raw.githubusercontent.com/google-research/google-research/master/collaborative_tr_collection/README.md), with workflow interpretation from [paper section 2.2](https://arxiv.org/html/2410.11056v1): `ref-Google` is newly commissioned from-scratch **human** translation. `ref-Google-llmrefine` is human translation refined with Gemini. MT edited by humans remains a separate assisted workflow. The archive has 1,976 aligned Chinese-source segments across 11 variants; the paper's smaller evaluation subset should not be confused with full-file extraction counts.
- [Spanish idiom dataset](https://zenodo.org/records/18351134), [release metadata](https://zenodo.org/api/records/18351134): explicitly named Google Translate, DeepL Classic/NMT and GPT-4o English outputs. These are six idiom-focused sets, not broad-domain or historical-MT coverage. The spreadsheet's 0/1 ratings concern idiom correctness, not authorship. Three missing quality ratings stay null. Summary/footer rows are excluded. Release date is recorded separately from unknown generation dates.

For older Google translation coverage, [SCATE](https://github.com/ardate/SCATE) has Google Translate outputs collected in 2014 and 2017, spanning statistical and neural MT. **Its target language is Dutch**, so it is useful for multilingual/older-MT experiments but not an English-positive training set. It has not been downloaded into the pilot. [DivEMT](https://github.com/gsarti/divemt) also separates from-scratch human translation and post-editing of Google/mBART outputs; its targets are six non-English languages. [MT Metrics Eval](https://github.com/google-research/mt-metrics-eval) is a convenient route to additional WMT years and language pairs, retaining system and document metadata.

[Par3](https://github.com/katherinethai/par3) is now imported separately for long, explicitly named Google Translate controls, as described above. Its README describes 106 books and 16 source languages; the downloaded archive contains 113 raw book/volume entries, and the long-paragraph selection covers 77 entries and 12 source languages. Released paragraph rows remain intact and are never joined. Par3 is excluded from pretraining inputs, and individual English translation rights remain unverified.

Do not use WMT parallel **training** corpora as automatically labelled machine outputs. They are inputs for training translators and may contain human translation, web alignment noise or mixed workflows. Likewise, human ratings of an output do not turn its authorship into human authorship.

| Label | Meaning | Binary detector target |
| --- | --- | --- |
| `human_translation` | Documented from-scratch human translation or human-only editing of it | 0 |
| `machine_translation` | Explicit system-produced translation | 1 |
| `machine_assisted` | MT with human edits, or human text passed through model refinement | 1 under the current strict process policy |
| `human_candidate` | Human origin inferred from documentary source/date | null; requires stronger provenance review |

These are process labels. An editing workflow may leave a passage unchanged. Across the pilot, 5,955 records have normalized text that appears under both human and machine/assisted labels. Those records remain in the full release with `ambiguous_text=true` and are excluded from binary training/evaluation. A text-only detector cannot recover different production histories from identical text. The binary export also deduplicates same-label text and retains all equivalent record IDs.

All translations of the same source passage and every segment of a source document share a partition. Cross-dataset exact normalized source/target matches are also grouped transitively. This protects against exact-text leakage; near duplicates remain a separate job. The most extensive connected group spans many records, so split sizes are not exactly proportional.

## Evaluate a detector

Score only `binary_examples.jsonl`, using English `text` as the detector input. Source text is retained for audit/optional bilingual research, but should not be required by a detector intended to filter monolingual pretraining text. Keep metadata such as system names, dates, workflow labels and filenames out of the classifier features.

Supply JSONL with `id` and a numeric `score` between zero and one; higher scores mean more machine-like text. Select thresholds on validation data, then evaluate the frozen threshold on test data:

```bash
uv run python datawork.py evaluate \
  --scores /path/to/validation-scores.jsonl --threshold 0.95 --split validation
uv run python datawork.py evaluate \
  --scores /path/to/test-scores.jsonl --threshold 0.95 --split test
```

For large-span scoring, use IDs from `binary_spans.jsonl` and add `--examples data/pilot/binary_spans.jsonl` to the evaluation command.

The threshold above is illustrative, not calibrated. The report shows human false positives and machine recall by source, engine, workflow, domain, source language and passage length, with word-weighted variants and missing-score coverage. A score need not be a calibrated probability, despite the zero-to-one range. Simple Wilson intervals treat rows as independent and are preliminary: final uncertainty should use document-cluster resampling. Incomplete score coverage must not be presented as a full-test result.

Include original English, human translations and varied English writers in the human controls. The current deduplicated detector set contains translated English controls only; Hansard candidates need provenance review before being added as controls. Add held-out engines, language pairs, years and production-like passage lengths. Current hash partitions test new documents, not unseen generators or domains.

To measure a claimed removal rate or retained contamination, audit both retained and rejected material using a source-stratified sampling design. Keep confirmed human, confirmed machine/translated, and unresolved cases separate. Report token-weighted outcomes with the actual tokenizer. The convenience pilot and a benchmark detector score cannot establish corpus-wide contamination.

## What remains for a production run

Review the publisher-stratified queue, establish authorship and captured-version evidence, and decide source proportions and an optional overall word cap. Add original-English human controls and independent long MT controls before calibrating any learned removal filter. NEWSROOM can be imported when its access files arrive. The current outputs preserve uncertainty rather than certify unknown text as human.
