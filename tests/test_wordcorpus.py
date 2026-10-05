import argparse
import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

import datawork as d
import newswork as nw
import wordcorpus


class WordCorpusTests(unittest.TestCase):
    def test_publisher_license_discrepancy_is_separate_from_original_metadata(self):
        row = {'dataset': 'fixture', 'publisher': 'publisher.example', 'license': 'CC-BY-4.0', 'metadata': {'license': 'CC-BY-4.0'}}
        inventory = {'datasets': [{'id': 'fixture', 'publisher_observations': {'publisher.example': {
            'status': 'license_version_mismatch_requires_reconciliation', 'publisher_observed_license': 'CC-BY-3.0'}}}]}
        wordcorpus.annotate_license(row, inventory)
        self.assertEqual(row['license'], 'CC-BY-4.0')
        self.assertEqual(row['metadata']['license'], 'CC-BY-4.0')
        self.assertEqual(row['publisher_license_observation']['publisher_observed_license'], 'CC-BY-3.0')
        self.assertEqual(row['license_review_status'], 'license_version_mismatch_requires_reconciliation')

    def fixture(self, root, rows):
        source = root / "curated"
        source.mkdir()
        path = source / "cc_news.spans.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        nw.save_json(source / "manifest.json", {"outputs": {path.name: nw.digest_file(path)}})
        nw.save_json(source / "verification.json", {"verified": True, "manifest_sha256": nw.digest_file(source / "manifest.json")})
        return source

    def test_word_cap_keeps_whole_spans_and_preserves_partitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for key, text in [("a", "one two three"), ("b", "four five six")]:
                row = d.record("cc_news", key, text, "human_candidate", key)
                row.update(binary_target=None, split="test", split_group=key, license="unknown")
                rows.append(row)
            source = self.fixture(root, rows)
            args = argparse.Namespace(curated=source, sources=["cc_news"], word_budget=5, output=root / "corpus", seed="test")
            with contextlib.redirect_stdout(io.StringIO()): wordcorpus.build(args)
            report = json.loads((root / "corpus/manifest.json").read_text())
            self.assertEqual(report["counts"]["selected_words"], 3)
            self.assertEqual(report["unfilled_words"], 2)
            self.assertEqual(report["rejections"]["exceeds_remaining_word_budget"], 1)
            with gzip.open(root / "corpus/test.jsonl.gz", "rt") as handle:
                row = json.loads(handle.readline())
            self.assertIn(row["text"], {r["text"] for r in rows})
            self.assertEqual(row["binary_target"], None)
            with self.assertRaises(ValueError): wordcorpus.build(args)

    def test_corrupted_span_archive_is_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root, [])
            with (source / "cc_news.spans.jsonl.gz").open("ab") as handle:
                handle.write(b"corrupted")
            args = argparse.Namespace(curated=source, sources=["cc_news"], word_budget=None, output=root / "corpus", seed="test")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                wordcorpus.build(args)
            self.assertFalse(args.output.exists())

    def test_exact_span_duplicates_cannot_cross_partitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [d.record("cc_news", key, "identical translated passage", "human_candidate", key) for key in ("a", "b")]
            for row, split in zip(rows, ("train", "test")):
                row.update(binary_target=None, split=split, split_group=row["id"], license="unknown")
            source = self.fixture(root, rows)
            args = argparse.Namespace(curated=source, sources=["cc_news"], word_budget=None, output=root / "corpus", seed="test")
            with self.assertRaisesRegex(ValueError, "crosses source partitions"):
                wordcorpus.build(args)


if __name__ == "__main__":
    unittest.main()
