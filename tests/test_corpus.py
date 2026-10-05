import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers
import corpus
import datawork


class CorpusTests(unittest.TestCase):
    def test_gigaword_selection_enforces_real_token_budget_eos_date_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixture.sgm"
            # Synthetic fixtures are exclusively for tests; never corpus inputs.
            docs = [("NYT_ENG_19980101.0001", "story", "alpha beta gamma"),
                    ("NYT_ENG_19980101.0002", "story", "alpha beta gamma"),
                    ("NYT_ENG_20240101.0001", "story", "future"),
                    ("NYT_ENG_19980101.0003", "other", "table"),
                    ("NYT_ENG_19980101.0004", "story", "delta epsilon zeta")]
            source.write_text("".join(f'<DOC id="{key}" type="{kind}">\n<TEXT>\n<P>\n{text}\n</P>\n</TEXT>\n</DOC>\n'
                                       for key, kind, text in docs))
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            args = argparse.Namespace(input=[f"gigaword={source}"], quota=[], token_budget=8,
                                      evidence="documented", cutoff="2021-12-31", pile_snapshot_date=None,
                                      tokenizer_json=tokenizer_path, output=root / "out", min_document_tokens=1,
                                      eos_token_id=1, seed="fixture")
            with contextlib.redirect_stdout(io.StringIO()): corpus.select(args)
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(manifest["selected_tokens"], 8)
            self.assertTrue(manifest["budget_met_exactly"])
            self.assertEqual(manifest["documents_by_source"], {"gigaword": 2})
            self.assertEqual(manifest["rejections"]["gigaword:duplicate_id_or_normalized_text"], 1)
            self.assertEqual(manifest["rejections"]["gigaword:publication_after_cutoff"], 1)
            self.assertEqual(manifest["rejections"]["gigaword:non_story"], 1)
            rows = sum((datawork.read_jsonl(root / f"out/{split}.jsonl") for split in ["train", "validation", "test"]), [])
            self.assertTrue(all(r["token_count"] == 4 for r in rows))
            with self.assertRaises(ValueError): corpus.select(args)

    def test_gigaword_keeps_body_entities_and_paragraphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.sgm"
            path.write_text('<DOC id="APW_ENG_20000101.0001" type="story">\n<HEADLINE>Headline</HEADLINE>\n'
                            '<TEXT>\n<P>First &amp; second.</P>\n<P>Next paragraph.</P>\n</TEXT>\n</DOC>\n')
            rows = list(corpus.gigaword_rows(path, __import__("datetime").date(2021, 12, 31)))
            self.assertEqual(rows[0][0]["text"], "First & second.\n\nNext paragraph.")

    def test_pile_excludes_synthetic_math_and_requires_allowlisted_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.jsonl"
            datawork.write_jsonl(path, [{"text": "Generated question", "meta": {"pile_set_name": "DM Mathematics"}},
                                       {"text": "Unknown prose", "meta": {"pile_set_name": "Pile-CC"}},
                                       {"text": "Court prose", "meta": {"pile_set_name": "FreeLaw"}}])
            date = __import__("datetime").date
            rows = list(corpus.jsonl_rows(path, "pile", date(2021, 12, 31), date(2020, 12, 31)))
            self.assertIsNone(rows[0][0]); self.assertIsNone(rows[1][0])
            self.assertEqual(rows[2][0]["source"], "FreeLaw")
            self.assertEqual(rows[2][0]["authorship"], "human_candidate")


if __name__ == "__main__":
    unittest.main()
