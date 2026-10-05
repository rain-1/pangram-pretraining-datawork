import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare as p


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.release = self.root / "release"
        self.release.mkdir()
        self.output = self.root / "output"
        self.body = "alpha " * 450 + "\n" + "beta " * 450
        (self.cache / "book.txt").write_bytes(self.body.encode())
        parent = {"id": "book/1", "dataset": "gutenberg_selected", "split": "train",
                  "split_group": "book/1", "text_sha256": p.text_hash(self.body),
                  "whitespace_words": 900, "language": "en", "authorship": "human_candidate",
                  "binary_target": None}
        rows = []
        for i, (a, b) in enumerate(((0, 2700), (2701, len(self.body)))):
            text = self.body[a:b]
            rows.append({"id": f"span/{i}", "parent_document_id": parent["id"],
                         "split": "train", "split_group": "book/1", "text_sha256": p.text_hash(text),
                         "whitespace_words": 450, "character_range": [a, b], "language": "en",
                         "authorship": "human_candidate", "binary_target": None})
        recipe = {"schema_version": 1, "release": "fixture", "original_manifest_sha256": "original",
                  "selection_sha256": p.selection_hash(rows), "passages": 2, "words": 900,
                  "parent_documents": 1, "summary": {"by_split": {"train": {"passages": 2, "words": 900},
                    "validation": {"passages": 0, "words": 0}, "test": {"passages": 0, "words": 0}}},
                  "notes": [], "records": rows, "parents": [{"metadata": parent, "selector": {
                      "kind": "book", "cache_path": "book.txt", "range": [0, len(self.body)]}}],
                  "sources": [{"cache_path": "book.txt", "kind": "file", "url": "https://example.test/book",
                               "sha256": p.digest_file(self.cache / "book.txt"),
                               "bytes": len(self.body.encode())}]}
        self.recipe = recipe
        (self.release / "recipe.json").write_bytes(p.json_bytes(recipe))
        (self.release / "licenses.snapshot.json").write_text("{}\n")
        (self.release / "ATTRIBUTION.txt").write_text("Fixture attribution\n")
        lock = {"files": {n: p.digest_file(self.release / n)
                          for n in ("recipe.json", "licenses.snapshot.json", "ATTRIBUTION.txt")}}
        (self.release / "release.json").write_bytes(p.json_bytes(lock))

    def build(self, output=None):
        return p.prepare(output or self.output, self.cache, offline=True, release=self.release)

    def test_replay_exact_text_and_idempotent_without_cache(self):
        self.build()
        actual = p.read_rows(self.output / "train.jsonl.gz")
        self.assertEqual([r["text"] for r in actual], ["alpha " * 450, "beta " * 450])
        (self.cache / "book.txt").unlink()
        with patch.object(p, "fetch_source", side_effect=AssertionError("Unexpected fetch")):
            self.assertTrue(self.build()["verified"])

    def test_reproducible_bytes_in_other_directory(self):
        self.build()
        other = self.root / "other"
        self.build(other)
        for source in self.output.iterdir():
            self.assertEqual(source.read_bytes(), (other / source.name).read_bytes())

    def test_corrupt_output_is_preserved_and_rejected(self):
        self.build()
        path = self.output / "train.jsonl.gz"
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "Export changed"):
            self.build()
        self.assertEqual(path.read_bytes(), b"corrupted")

    def test_corrupt_cached_input_is_preserved_and_rejected(self):
        path = self.cache / "book.txt"
        path.write_text("changed source")
        with self.assertRaisesRegex(ValueError, "Pinned source differs"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(path.read_text(), "changed source")

    def test_upstream_mismatch_leaves_no_cached_or_partial_file(self):
        spec = {**self.recipe["sources"][0], "cache_path": "download.txt"}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"changed upstream")):
            with self.assertRaisesRegex(ValueError, "Pinned source differs"):
                p.fetch_source(spec, self.cache)
        self.assertFalse((self.cache / "download.txt").exists())
        self.assertEqual(list(self.cache.glob("*.part")), [])

    def test_download_keeps_only_complete_pinned_prefix_rows(self):
        prefix = b'{"id": 1}\n{"id": 2}\n'
        spec = {"kind": "gzip_jsonl_prefix", "cache_path": "prefix.jsonl", "url": "https://example.test",
                "rows": 2, "sha256": p.text_hash(prefix.decode()), "bytes": len(prefix)}
        payload = gzip.compress(prefix + b'{"id": 3}\n')
        with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
            p.fetch_source(spec, self.cache)
        self.assertEqual((self.cache / "prefix.jsonl").read_bytes(), prefix)

    def test_group_leakage_rejected_even_when_identity_matches(self):
        rows = p.reconstruct_records(self.recipe, p.reconstruct_parents(self.recipe, self.cache))
        rows[1]["split"] = "test"
        recipe = {**self.recipe, "selection_sha256": p.selection_hash(rows)}
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            p.check_records(rows, recipe)

    def test_invalid_filenames_and_slice_bounds_rejected(self):
        for name in ("../secret", "/absolute", ".", "..", "folder\\file"):
            with self.assertRaises(ValueError):
                p.plain_name(name)
        for value in ([-1, 5], [0, 99], [5, 5], [True, 8]):
            with self.assertRaises(ValueError):
                p.checked_range(value, 10)


if __name__ == "__main__":
    unittest.main()
