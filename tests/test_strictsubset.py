import copy
import json
from pathlib import Path
import unittest

import datawork as d
import strictsubset as s


class StrictSubsetTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((Path(__file__).resolve().parents[1] / "config/human_core.json").read_text())
        text = "Lord Smith: asked Her Majesty's Government:\nWhat plans they have for public services.\n\nBaroness Jones: " + "The department will consult local communities about improving public services. " * 50
        self.row = {"id": "span/a", "dataset": "hansard", "source": "uk-hansard-lords-written-answers", "publisher": "UK Parliament",
                    "language": "en", "metadata": {"language": "en", "year": "2005"}, "publication_date": "2005-06-01", "original_id": "lordswrans2005-06-01a",
                    "license": "Open Parliament Licence - " + s.LICENSE_URL, "authorship": "human_candidate", "binary_target": None,
                    "text": text, "whitespace_words": len(text.split()), "text_sha256": d.text_hash(text),
                    "parent_document_id": "a", "character_range": [0, len(text)], "split": "train", "split_group": "a"}
        self.parent = copy.deepcopy(self.row)
        self.parent["id"] = "a"

    def test_accepts_historical_named_prose_without_certifying_labels(self):
        before = copy.deepcopy(self.row)
        reason, evidence = s.decision(self.row, {"a": self.parent}, self.policy)
        self.assertIsNone(reason)
        self.assertFalse(evidence["individual_authorship_verified"])
        self.assertEqual(before, self.row)
        self.assertIsNone(self.row["binary_target"])

    def test_old_publication_does_not_override_new_captured_version(self):
        self.row["text_version_date"] = "2024-06-01"
        self.assertEqual(s.source_gate(self.row, self.policy), "known_text_version_after_strict_cutoff")
        self.row.pop("text_version_date")
        self.row["metadata"]["year"] = "2006"
        self.assertEqual(s.source_gate(self.row, self.policy), "date_id_year_disagreement")

    def test_machine_notice_elsewhere_in_parent_quarantines_passage(self):
        self.parent["text"] += "\nThe appendix was translated using Google Translate."
        reason, _ = s.decision(self.row, {"a": self.parent}, self.policy)
        self.assertEqual(reason, "parent_generation_or_translation_mention_requires_review")

    def test_deeply_is_not_deepl(self):
        self.assertIsNone(s.NOTICE.search("We are deeply concerned about access to services."))
        self.assertIsNotNone(s.NOTICE.search("Translated by DeepL."))

    def test_extraction_annotation_elsewhere_in_parent_excludes_passage(self):
        self.parent["text"] += "\nQuestion number missing in Hansard, possibly truncated question."
        self.assertEqual(s.decision(self.row, {"a": self.parent}, self.policy)[0], "parent_extraction_annotation_requires_review")

    def test_named_headings_without_question_answer_are_insufficient(self):
        text = self.row["text"].replace("asked Her Majesty's Government:", "will make a statement.")
        self.row.update(text=text, text_sha256=d.text_hash(text), whitespace_words=len(text.split()), character_range=[0, len(text)])
        self.parent["text"] = text
        self.assertEqual(s.decision(self.row, {"a": self.parent}, self.policy)[0], "no_explicit_named_question_and_answer")

    def test_tabular_content_is_screened_even_with_person_attributions(self):
        text = self.row["text"] + "\n" + "\n".join("\t\tDistrict 123 456" for _ in range(20))
        self.row.update(text=text, text_sha256=d.text_hash(text), whitespace_words=len(text.split()), character_range=[0, len(text)])
        self.parent["text"] = text
        self.assertTrue(s.decision(self.row, {"a": self.parent}, self.policy)[0].startswith("prose_screen:"))

    def test_text_changes_and_parent_split_changes_are_rejected(self):
        self.parent["text"] = "Different " + self.parent["text"]
        with self.assertRaisesRegex(ValueError, "body or partition"):
            s.decision(self.row, {"a": self.parent}, self.policy)
        self.parent = copy.deepcopy(self.row)
        self.parent["split"] = "test"
        with self.assertRaisesRegex(ValueError, "body or partition"):
            s.decision(self.row, {"a": self.parent}, self.policy)


if __name__ == "__main__":
    unittest.main()
