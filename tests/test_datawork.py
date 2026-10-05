import csv
from pathlib import Path
import tempfile
import unittest

import datawork as d


class DataworkTests(unittest.TestCase):
    def hansard(self, **updates):
        row = {"id": "lordswrans2016-06-15", "source": "uk-hansard-lords-written-answers",
               "added": "2024-06-01", "text": "An official answer.",
               "metadata": {"language": "en", "year": "2016"}}
        row.update(updates)
        return row

    def test_hansard_uses_document_date_and_does_not_treat_ingestion_as_authorship(self):
        self.assertEqual(d.hansard_decision(self.hansard(), 2021), ("retain_candidate", "2016-06-15"))
        self.assertEqual(d.hansard_decision(self.hansard(id="lordswrans2023-06-15",
                             metadata={"language": "en", "year": "2023"}), 2021)[0], "publication_after_cutoff")

    def test_hansard_fails_closed_for_unreliable_date_and_translation_channel(self):
        for row, reason in [
            (self.hansard(id="unknown"), "missing_document_date"),
            (self.hansard(id="lordswrans2016-02-30"), "invalid_date_or_year"),
            (self.hansard(metadata={"language": "en", "year": "2015"}), "date_year_disagreement"),
            (self.hansard(metadata={"language": "cy", "year": "2016"}), "language_not_explicitly_english"),
            (self.hansard(source="uk-hansard-senedd-debates"), "source_requires_translation_provenance_review"),
        ]:
            with self.subTest(reason=reason):
                self.assertEqual(d.hansard_decision(row, 2021)[0], reason)

    def test_mqm_deduplicates_ratings_removes_wrappers_and_quarantines_disagreement(self):
        fields = ["system", "doc", "seg_id", "source", "target"]
        rows = [
            ["ref.A", "doc1", "1", "source <v>one</v>", "A <v>human</v> sentence."],
            ["ref.A", "doc1", "1", "source one", "A human sentence."],
            ["hyp.engine", "doc1", "1", "source one", "Machine version."],
            ["hyp.engine", "doc1", "1", "source one", "A different machine version."],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.tsv"
            with path.open("w", newline="") as f:
                w = csv.writer(f, delimiter="\t"); w.writerow(fields); w.writerows(rows)
            parsed, rejected = d.parse_wmt21(path, "fixture")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["text"], "A human sentence.")
        self.assertEqual(parsed[0]["source_text"], "source one")
        self.assertEqual(parsed[0]["annotation_rows"], 2)
        self.assertEqual(rejected[0]["reason"], "conflicting_annotation_text")

    def test_google_commissioned_reference_is_human_and_llm_refinement_is_assisted(self):
        self.assertEqual(d.COLLABORATIVE["ref-Google"][0], "human_translation")
        self.assertEqual(d.COLLABORATIVE["ref-Google-llmrefine"][0], "machine_assisted")
        self.assertEqual(d.COLLABORATIVE["ref-mid-wmt-edited"][0], "machine_assisted")

    def test_splits_keep_documents_translation_variants_and_cross_dataset_duplicates_together(self):
        rows = [
            d.record("a", "1", "Human wording.", "human_translation", "one", source_text="Chinese source", source_language="zh"),
            d.record("a", "2", "Machine wording.", "machine_translation", "one", source_text="Chinese source", source_language="zh"),
            d.record("b", "3", "Other wording.", "machine_translation", "two", source_text="Chinese source", source_language="zh"),
            d.record("c", "4", "HUMAN   WORDING.", "human_translation", "three"),
        ]
        d.assign_splits(rows); d.verify_records(rows)
        self.assertEqual(len({r["split_group"] for r in rows}), 1)
        self.assertEqual(len({r["split"] for r in rows}), 1)
        expected = {r["id"]: (r["split"], r["split_group"]) for r in rows}
        d.assign_splits(list(reversed(rows)))
        self.assertEqual(expected, {r["id"]: (r["split"], r["split_group"]) for r in rows})

    def test_identical_text_with_conflicting_authorship_has_no_binary_target(self):
        rows = [d.record("a", "1", "Same text.", "human_translation", "one"),
                d.record("a", "2", "same text.", "machine_translation", "two"),
                d.record("a", "3", "Candidate prose.", "human_candidate", "three")]
        d.assign_splits(rows)
        self.assertTrue(rows[0]["ambiguous_text"])
        self.assertIsNone(rows[0]["binary_target"])
        self.assertIsNone(rows[1]["binary_target"])
        self.assertIsNone(rows[2]["binary_target"])

    def test_binary_export_deduplicates_same_origin_and_preserves_equivalent_ids(self):
        rows = [d.record("a", "1", "Human text.", "human_translation", "one"),
                d.record("a", "2", "human text.", "human_translation", "two"),
                d.record("a", "3", "Candidate text.", "human_candidate", "three")]
        d.assign_splits(rows)
        examples = d.binary_examples(rows)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["equivalent_record_ids"], ["a/1", "a/2"])

    def test_translation_spans_keep_common_boundaries_and_do_not_bridge_missing_segments(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        rows = []
        for index, human, machine in [(1, "a b", "e f"), (2, "c d", "g h"), (4, "i j", "k l")]:
            for engine, label, text in [("ref.A", "human_translation", human), ("hyp.mt", "machine_translation", machine)]:
                rows.append(d.record("fixture", f"{index}/{engine}", text, label, "one",
                                     source_text=f"source {index}", source_language="zh", engine=engine, segment_index=index))
        d.assign_splits(rows)
        spans, skipped = d.translation_spans(rows, tokenizer, 3, 4)
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(row["span_segment_range"] == [1, 2] for row in spans))
        self.assertTrue(all(row["token_count"] == 4 for row in spans))
        self.assertEqual(skipped["aligned_block_too_short"], 1)

    def test_verifier_catches_corruption_and_duplicate_leakage(self):
        rows = [d.record("a", "1", "Identical prose.", "human_translation", "one"),
                d.record("a", "2", "Identical prose.", "human_translation", "two")]
        d.assign_splits(rows)
        rows[1]["split"] = "test" if rows[0]["split"] != "test" else "train"
        with self.assertRaises(ValueError): d.verify_records(rows)
        rows[1]["split"] = rows[0]["split"]
        rows[0]["text"] = "Changed text."
        with self.assertRaises(ValueError): d.verify_records(rows)

    def scored_fixture(self):
        rows = [d.record("fixture", "h1", "Human one.", "human_translation", "h1"),
                d.record("fixture", "h2", "Human two.", "human_translation", "h2"),
                d.record("fixture", "m1", "Machine one.", "machine_translation", "m1"),
                d.record("fixture", "m2", "Machine two.", "machine_translation", "m2")]
        d.assign_splits(rows)
        for row in rows: row["split"] = "test"
        return rows

    def test_evaluation_reports_fpr_recall_missing_coverage_and_not_contamination(self):
        rows = self.scored_fixture()
        scores = [{"id": r["id"], "score": s} for r, s in zip(rows, [0.1, 0.8, 0.9, 0.2])]
        report = d.evaluate(rows, scores, 0.5, "test")
        metrics = report["metrics"]["all"]
        self.assertEqual(metrics["confusion"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(metrics["human_false_positive_rate"], 0.5)
        self.assertEqual(metrics["machine_recall"], 0.5)
        self.assertEqual(d.evaluate(rows, scores[:-1], 0.5, "test")["missing_eligible_scores"], 1)

    def test_evaluation_rejects_nan_duplicate_unknown_and_out_of_range_scores(self):
        rows = self.scored_fixture()
        for scores in [[{"id": rows[0]["id"], "score": float("nan")}],
                       [{"id": rows[0]["id"], "score": 1.1}],
                       [{"id": "unknown", "score": 0.5}],
                       [{"id": rows[0]["id"], "score": 0.5}] * 2]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                d.evaluate(rows, scores, 0.5, "test")


if __name__ == "__main__":
    unittest.main()
