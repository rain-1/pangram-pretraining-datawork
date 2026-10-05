import collections
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import par3work
import prepare as p
import prepare_translations as pt


class TranslationSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.release = self.root / "release"
        self.release.mkdir()
        self.output = self.root / "output"
        data = {"book_fr": {"source_paras": ["a foreign source paragraph"],
                           "gt_paras": ["machine " * 450],
                           "translator_data": {"translator_1": {"translator_paras": ["human " * 450]}}}}
        (self.cache / "par3.pkl").write_bytes(pickle.dumps(data))
        self.config = {"repo": "fixture", "revision": "fixture", "minimum_words": 400,
                       "maximum_words": 3000, "download_url": "https://example.test/archive",
                       "sha256": p.digest_file(self.cache / "par3.pkl"),
                       "bytes": (self.cache / "par3.pkl").stat().st_size}
        self.records, self.binary, _ = par3work.extract(data, self.config, self.config["sha256"])
        identity = pt.identities(self.binary)
        self.selection = {"schema_version": 1, "release": "fixture", "original_manifest_sha256": "original",
                          "selection_sha256": pt.identity_hash(identity), "identities": identity,
                          "binary_examples": len(self.binary),
                          "by_split_label": {s: dict(collections.Counter(r["authorship"] for r in self.binary
                                                                          if r["split"] == s)) for s in p.SPLITS},
                          "by_source_language": {"fr": 2},
                          "independent_work_groups_per_split": {"train": 1, "validation": 0, "test": 0}}
        (self.release / "selection.json").write_bytes(p.json_bytes(self.selection))
        (self.release / "source.json").write_bytes(p.json_bytes(self.config))
        (self.release / "licenses.snapshot.json").write_text("{}\n")
        self.lock = {"files": {n: p.digest_file(self.release / n)
                              for n in ("selection.json", "source.json", "licenses.snapshot.json")}}
        (self.release / "release.json").write_bytes(p.json_bytes(self.lock))

    def build(self):
        return pt.prepare(self.output, self.cache, offline=True, release=self.release)

    def test_exact_controls_and_rerun_without_archive(self):
        report = self.build()
        self.assertEqual(report["labels"], {"machine_translation": 1, "human_translation": 1})
        actual = [json.loads(x) for x in (self.output / "binary_examples.jsonl").read_text().splitlines()]
        self.assertEqual(pt.identities(actual), self.selection["identities"])
        (self.cache / "par3.pkl").unlink()
        with patch.object(p, "fetch_source", side_effect=AssertionError("Unexpected fetch")):
            self.assertTrue(self.build()["verified"])

    def test_bad_cached_archive_is_not_overwritten(self):
        path = self.cache / "par3.pkl"
        path.write_bytes(b"bad archive")
        with self.assertRaisesRegex(ValueError, "Pinned source differs"):
            self.build()
        self.assertEqual(path.read_bytes(), b"bad archive")
        self.assertFalse(self.output.exists())

    def test_corrupt_export_is_not_overwritten(self):
        self.build()
        path = self.output / "binary_examples.jsonl"
        path.write_text("corrupt export")
        with self.assertRaisesRegex(ValueError, "Translation export changed"):
            self.build()
        self.assertEqual(path.read_text(), "corrupt export")

    def test_foreign_source_change_is_detected(self):
        for row in self.records + self.binary:
            row["source_text"] = "a different source"
        with self.assertRaisesRegex(ValueError, "Selected controls"):
            pt.check_examples(self.records, self.binary, self.selection, self.config)

    def test_wrong_process_label_cannot_match_frozen_selection(self):
        row = next(r for r in self.binary if r["binary_target"] == 1)
        row["engine"] = "Different generator"
        with self.assertRaises(ValueError):
            pt.check_examples(self.records, self.binary, self.selection, self.config)


if __name__ == "__main__":
    unittest.main()
