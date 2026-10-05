import datetime as dt
import json
from pathlib import Path
import unittest

import curate


class CurationTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((Path(curate.ROOT) / "config/curation.json").read_text())

    def test_old_publication_does_not_hide_recent_revision(self):
        row = {"created": "2 February 2016 at 8:05 (Updated on 11 January 2024 at 1:37)",
               "source": "news-mekongeye", "metadata": {"url": "https://mekongeye.com/2016/02/02/example/"}}
        date, updated, _, reason = curate.news_dates(row, "common_pile_news", self.policy)
        self.assertEqual((date, updated, reason), (dt.date(2016, 2, 2), dt.date(2024, 1, 11), None))
        row.update(text="English news prose. " * 150)
        row["metadata"]["license"] = "https://creativecommons.org/licenses/by/4.0/"
        self.assertEqual(curate.preliminary(row, "common_pile_news", self.policy)[1], "explicit_update_after_cutoff")

    def test_dated_urls_are_source_specific_and_conflicts_fail_closed(self):
        row = {"created": None, "source": "news-globalvoices",
               "metadata": {"url": "https://globalvoices.org/2004/12/04/example/"}}
        self.assertEqual(curate.news_dates(row, "common_pile_news", self.policy)[0], dt.date(2004, 12, 4))
        row["created"] = "December 5, 2004"
        self.assertEqual(curate.news_dates(row, "common_pile_news", self.policy)[3], "publication_date_url_conflict")
        row.update(source="news-altnews", created=None)
        self.assertEqual(curate.news_dates(row, "common_pile_news", self.policy)[3], "missing_or_unparsed_publication_date")
        row["metadata"]["url"] = None
        self.assertEqual(curate.news_dates(row, "common_pile_news", self.policy)[3], "missing_or_unparsed_publication_date")

    def test_ordinals_and_invalid_dates_are_not_silently_reinterpreted(self):
        self.assertEqual(curate.explicit_dates("10th April 2017 / Last updated: 12th July 2018"),
                         [dt.date(2017, 4, 10), dt.date(2018, 7, 12)])
        self.assertEqual(curate.explicit_dates("February 30, 2018"), [])
        row = {"created": "December 5, 2004 updated yesterday", "metadata": {}}
        self.assertEqual(curate.news_dates(row, "common_pile_news", self.policy)[3], "unparsed_explicit_update_date")

    def test_outbound_text_does_not_inherit_host_license(self):
        row = {"text": "Example news story. " * 150, "created": "January 1, 2017",
               "source": "news-milwaukeenns", "metadata": {"url": "https://different-publisher.example/article/",
               "license": "https://creativecommons.org/licenses/by/4.0/"}}
        self.assertEqual(curate.preliminary(row, "common_pile_news", self.policy)[1], "source_host_mismatch_or_unreviewed_source")

    def test_human_translation_stays_eligible_machine_notice_is_quarantined(self):
        row = {"text": "This article was translated by a volunteer. " + "The reporter described the event. " * 100,
               "created": "January 1, 2017", "source": "news-globalvoices",
               "metadata": {"url": "https://globalvoices.org/2017/01/01/example/",
               "license": "https://creativecommons.org/licenses/by/4.0/"}}
        self.assertIsNone(curate.preliminary(row, "common_pile_news", self.policy)[1])
        row["text"] = "This article was automatically translated. " + row["text"]
        self.assertEqual(curate.preliminary(row, "common_pile_news", self.policy)[1], "explicit_machine_translation_notice_requires_review")

    def test_spans_are_exact_contiguous_unicode_slices_bounded_by_words(self):
        text = "  " + "\n\n".join("Café 東京 😀 punctuation. " * 13 for _ in range(12)) + " trailing short tail"
        spans = curate.make_spans(text, 30, 120)
        self.assertGreater(len(spans), 1)
        end_previous = 0
        for start, end, words in spans:
            self.assertEqual(start, end_previous)
            self.assertEqual(words, len(text[start:end].split()))
            self.assertTrue(30 <= words <= 120)
            self.assertFalse(text[start:end].startswith("afé"))
            end_previous = end
        self.assertLess(len(text[end_previous:].split()), 30)
        self.assertEqual(curate.make_spans("short", 30, 120), [])
        with self.assertRaises(ValueError):
            curate.make_spans(text, 120, 30)

    def test_tracking_parameters_merge_but_article_identity_parameters_remain(self):
        self.assertEqual(curate.canonical_url("http://www.bbc.com/article/?utm_source=test&id=4#top"),
                         "https://bbc.com/article?id=4")
        self.assertNotEqual(curate.canonical_url("https://bbc.com/?id=4"), curate.canonical_url("https://bbc.com/?id=5"))

    def test_eligible_final_tail_is_not_divided_into_ineligible_pieces(self):
        # A long final sentence follows a short sentence with very long words.
        # A preferred punctuation boundary must not discard both short pieces.
        text = "first " * 120 + "\n\n" + ("internationalization " * 29) + ". " + ("x " * 30)
        spans = curate.make_spans(text, 40, 120)
        self.assertEqual(sum(n for _, _, n in spans), len(text.split()))
        self.assertEqual(spans[-1][1], len(text))


if __name__ == "__main__":
    unittest.main()
