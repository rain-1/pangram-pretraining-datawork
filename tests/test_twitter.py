import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare as p
import prepare_grok as grok
import prepare_twitter as pt
import twitterwork as tw


class TwitterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.release = self.root / "release"
        self.release.mkdir()
        self.config = {"schema_version": 1, "release": "twitter-v1", "sources": []}
        self.spec = {"dataset": "tweepfake", "cache_path": "tweepfake-train.csv", "split": "train",
                     "revision": "fixture", "kind": "file", "url": "https://example.invalid/unused"}
        with (self.cache / self.spec["cache_path"]).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["screen_name", "text", "account.type", "class_type"])
            for i in range(6):
                writer.writerow([f"human{i}", f"Human {i}; with a quoted\nnewline", "human", "human"])
                writer.writerow([f"bot{i}", f"Generated {i}", "bot", "gpt2"])
            writer.writerow(["unknown", "Unknown technology", "bot", "others"])
        self.add_spec(self.spec)
        for model in ("modelA", "modelB"):
            spec = {"dataset": "unmasking", "cache_path": f"{model}.json", "split": "train",
                    "model": model, "revision": "fixture", "kind": "file", "url": "https://example.invalid/unused"}
            rows = [{"text": "A shared human reference", "label": 0},
                    {"text": f"A generated response from {model}", "label": 1}]
            (self.cache / spec["cache_path"]).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.add_spec(spec)

    def add_spec(self, spec):
        path = self.cache / spec["cache_path"]
        spec.update(bytes=path.stat().st_size, sha256=p.digest_file(path))
        self.config["sources"].append(spec)

    def freeze(self):
        result = tw.extract(self.config, self.cache)
        selection = tw.summary(*result)
        for name, value in (("source.json", self.config), ("selection.json", selection),
                            ("licenses.snapshot.json", {"datasets": []})):
            (self.release / name).write_bytes(p.json_bytes(value))
        lock = {"files": {n: p.digest_file(self.release / n) for n in
                          ("source.json", "selection.json", "licenses.snapshot.json")}}
        (self.release / "release.json").write_bytes(p.json_bytes(lock))
        return result

    def test_semicolon_csv_and_original_text(self):
        rows = tw.read_tweepfake(self.cache / self.spec["cache_path"], self.spec)
        self.assertEqual(rows[0]["text"], "Human 0; with a quoted\nnewline")
        self.assertEqual(rows[-1]["binary_target"], None)
        self.assertFalse(rows[0]["human_authorship_verified"])

    def test_dedup_pairs_and_account_splits(self):
        rows, binary, quarantine = tw.extract(self.config, self.cache)
        refs = [r for r in binary if r["text"] == "A shared human reference"]
        self.assertEqual(len(refs), 1)
        self.assertEqual(len(refs[0]["equivalent_record_ids"]), 2)
        paired = [r for r in rows if r["dataset"] == "unmasking"]
        self.assertEqual(len({r["split_group"] for r in paired}), 1)
        self.assertEqual(len({r["split"] for r in paired}), 1)
        self.assertEqual({r["split"] for r in binary}, set(p.SPLITS))
        self.assertEqual(quarantine[0]["exclusion_reason"], "unknown_bot_method")

    def test_same_text_merges_accounts_and_conflicts_are_excluded(self):
        rows, _, _ = tw.extract(self.config, self.cache)
        a = rows[0]
        b = next(r for r in rows if r["binary_target"] != a["binary_target"] and r["binary_target"] is not None)
        b.update(text=a["text"].upper(), text_sha256=p.text_hash(a["text"].upper()),
                 normalized_text_sha256=a["normalized_text_sha256"], whitespace_words=a["whitespace_words"])
        tw.assign_groups(rows)
        binary, quarantine = tw.deduplicate(rows)
        self.assertEqual(a["split_group"], b["split_group"])
        self.assertFalse(any(r["normalized_text_sha256"] == a["normalized_text_sha256"] for r in binary))
        self.assertTrue(any(r["exclusion_reason"] == "conflicting_labels_for_identical_normalized_text" for r in quarantine))
        tw.validate(rows, binary, quarantine)

    def test_unmasking_alternation_and_reference_consistency(self):
        spec = self.config["sources"][1]
        path = self.cache / spec["cache_path"]
        path.write_text('{"text":"bad","label":1}\n{"text":"wrong","label":0}\n')
        with self.assertRaisesRegex(ValueError, "alternation"):
            tw.read_unmasking(path, spec)
        path.write_text('{"text":"Different human","label":0}\n{"text":"generated","label":1}\n')
        spec.update(bytes=path.stat().st_size, sha256=p.digest_file(path))
        with self.assertRaisesRegex(ValueError, "sequence differs"):
            tw.extract(self.config, self.cache)

    def test_end_to_end_offline_idempotence_and_export_tamper(self):
        self.freeze()
        output = self.root / "ready"
        first = pt.prepare(output, self.cache, offline=True, release=self.release)
        second = pt.prepare(output, self.cache, offline=True, release=self.release)
        self.assertEqual(first, second)
        with (output / "test.jsonl.gz").open("ab") as f:
            f.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "export changed"):
            pt.prepare(output, self.cache, verify_only=True, release=self.release)

    def test_source_tamper_preserves_inputs_and_publishes_nothing(self):
        self.freeze()
        source = self.cache / self.spec["cache_path"]
        source.write_text("changed")
        output = self.root / "ready"
        with self.assertRaisesRegex(ValueError, "Pinned source differs"):
            pt.prepare(output, self.cache, offline=True, release=self.release)
        self.assertEqual(source.read_text(), "changed")
        self.assertFalse(output.exists())

    def test_release_metadata_tamper(self):
        self.freeze()
        (self.release / "selection.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "release changed"):
            pt.load_release(self.release)

    def test_validator_detects_leakage(self):
        rows, binary, quarantine = tw.extract(self.config, self.cache)
        pair = [r for r in rows if r["dataset"] == "unmasking"]
        pair[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "leakage"):
            tw.validate(rows, binary, quarantine)


class GrokTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "metadata"
        self.output.mkdir()
        def tweet(ident, role=True, lang="en", reply="900"):
            return {"id": ident, "inReplyToId": reply, "lang": lang, "isMediaOnly": False,
                    "author": {"isAssistant": role}, "createdAt": "2025 fixture"}
        self.tweet = tweet
        self.conversations = [
            {"conversationId": "100", "annotations": {"generated_summary": "Never tweet text"},
             "threads": [{"threadId": "10", "tweets": [tweet("1"), tweet("2", False), tweet("3", lang="es")]}]},
            {"conversationId": "101", "threads": [{"threadId": "11", "tweets": [tweet("1"), tweet("4")]}]},
            {"conversationId": "102", "threads": [{"threadId": "12", "tweets": [tweet("5"), tweet("6", reply="")]}]},
        ]

    def index(self):
        source = self.root / "raw.json"
        source.write_text(json.dumps(self.conversations))
        return grok.index_metadata(source, self.output)

    def hydrated(self, items):
        path = self.root / "hydrated.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in items) + "\n")
        with grok.connect_readonly(self.output / "index.sqlite") as db:
            return grok.read_hydrated(path, db)

    def test_ids_only_index_discards_annotations_and_merges_conversations(self):
        counts = self.index()
        self.assertEqual(counts["unique_tweet_ids"], 6)
        self.assertEqual(counts["english_assistant_reply_ids"], 3)
        candidates = p.read_rows(self.output / "assistant_candidates.jsonl.gz")
        self.assertTrue(all("text" not in r and r["binary_target"] is None for r in candidates))
        self.assertEqual(candidates[0]["split_group"], candidates[1]["split_group"])
        with grok.connect_readonly(self.output / "index.sqlite") as db:
            self.assertNotIn("text", [r[1] for r in db.execute("PRAGMA table_info(tweets)")])

    def test_only_english_assistant_reply_text_and_no_human_negatives(self):
        self.index()
        rows, binary, rejected = self.hydrated([{"id": str(i), "original_text": f"Original {i}"} for i in range(1, 7)]
                                              + [{"id": "99", "original_text": "Unknown"}])
        self.assertEqual({r["tweet_id"] for r in binary}, {"1", "4", "5"})
        self.assertTrue(all(r["binary_target"] == 1 for r in binary))
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        self.assertEqual({r["reason"] for r in rejected},
                         {"participant_authorship_unresolved", "not_english_or_language_conflict", "not_a_text_assistant_reply", "id_not_in_pinned_release"})

    def test_duplicate_text_merges_otherwise_independent_conversation_groups(self):
        self.index()
        rows, binary, _ = self.hydrated([{"id": "1", "original_text": "Same text"},
                                        {"id": "5", "original_text": "SAME TEXT"}])
        self.assertEqual(len(binary), 1)
        self.assertEqual(len(binary[0]["equivalent_record_ids"]), 2)
        self.assertEqual(rows[0]["split_group"], rows[1]["split_group"])
        self.assertEqual(binary[0]["text"], "Same text")

    def test_conflicting_metadata_is_quarantined(self):
        self.conversations[1]["threads"][0]["tweets"][0]["author"]["isAssistant"] = False
        counts = self.index()
        self.assertEqual(counts["conflicting_metadata_ids"], 1)
        rows, binary, rejected = self.hydrated([{"id": "1", "original_text": "Raw original"}])
        self.assertFalse(binary)
        self.assertEqual(rejected[0]["reason"], "conflicting_release_metadata")

    def test_requires_explicit_original_text_and_unique_id(self):
        self.index()
        with self.assertRaisesRegex(ValueError, "requires id and original_text"):
            self.hydrated([{"id": "1", "text": "Could be generated annotation"}])
        with self.assertRaisesRegex(ValueError, "Repeated hydrated"):
            self.hydrated([{"id": "1", "original_text": "one"}, {"id": "1", "original_text": "two"}])

    def test_reposts_and_language_mismatch_are_rejected(self):
        self.index()
        _, binary, rejected = self.hydrated([{"id": "1", "original_text": "RT @user: copied"},
                                             {"id": "4", "original_text": "Raw text", "lang": "fr"}])
        self.assertFalse(binary)
        self.assertEqual(len(rejected), 2)

    def test_metadata_and_local_import_replay_and_tamper(self):
        source = self.root / "raw.json"
        source.write_text(json.dumps(self.conversations))
        spec = {"cache_path": "grokset-dehydrated.json", "sha256": p.digest_file(source),
                "bytes": source.stat().st_size, "kind": "file", "revision": "fixture"}
        release = self.root / "release"
        release.mkdir()
        (release / "licenses.snapshot.json").write_text('{"datasets":[]}\n')
        lock = {"files": {"licenses.snapshot.json": p.digest_file(release / "licenses.snapshot.json")}}
        config = {"sources": [spec]}
        # Source acquisition is separately covered by pin/tamper tests. These
        # fixtures exercise the metadata and hydrated export publication paths.
        with patch.object(pt, "load_release", return_value=({}, config, lock)), \
                patch.object(pt, "RELEASE", release), patch.object(p, "fetch_source", return_value=source):
            first = grok.metadata(self.output, self.root, offline=True)
            self.assertEqual(first, grok.metadata(self.output, self.root, verify_only=True))
            input_path = self.root / "original.jsonl"
            input_path.write_text('{"id":"1","original_text":"Fixture original"}\n')
            output = self.root / "imported"
            imported = grok.import_text(input_path, self.output, output)
            self.assertEqual(imported["binary_examples"], 1)
            self.assertEqual(imported, grok.import_text(input_path, self.output, output))
            input_path.write_text('{"id":"1","original_text":"Changed original"}\n')
            with self.assertRaisesRegex(ValueError, "different inputs"):
                grok.import_text(input_path, self.output, output)
            with (self.output / "assistant_candidates.jsonl.gz").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                grok.metadata(self.output, self.root, verify_only=True)


if __name__ == "__main__":
    unittest.main()
