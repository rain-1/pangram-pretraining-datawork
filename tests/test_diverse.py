import copy
import json
from pathlib import Path
import unittest

import diverse


class DiverseTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((Path(__file__).resolve().parents[1] / "config/diverse_human.json").read_text())

    def test_wiki_articles_keep_sections_and_never_join_unrelated_titles(self):
        rows = [' = First = \n', '', 'Body one.\n', '', ' = = Section = = \n', '', 'More body.\n', '', ' = Second = \n', '', 'Other text.\n']
        articles = list(diverse.wiki_articles(rows))
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0][1:3], (0, 8))
        self.assertIn('Section', articles[0][3])
        self.assertNotIn('Other text', articles[0][3])

    def test_punctuation_restoration_is_mechanical_and_preserves_words(self):
        raw = "The game 's real @-@ time price was 1 @,@ 000 @.@ 5 dollars .\n"
        expected = "The game's real-time price was 1,000.5 dollars.\n"
        self.assertEqual(diverse.restore_punctuation(raw), expected)
        self.assertEqual(diverse.restore_punctuation(expected), expected)

    def test_table_key_equalities_do_not_become_fake_wikipedia_articles(self):
        rows = [' = Cricket season = \n', '', 'Key : Pld\n', ' = Played , W = \n', 'Wins , L\n', ' = Losses , D = \n', '', ' = = Results = = \n', '', 'Prose.\n', '', ' = Genuine article = \n', '', 'Other prose.\n']
        articles = list(diverse.wiki_articles(rows))
        self.assertEqual([r[0] for r in articles], ['Cricket season', 'Genuine article'])
        self.assertEqual(articles[0][1:3], (0, 11))

    def test_inline_math_equalities_stay_in_their_original_article(self):
        rows = [' = Planet = \n', '', 'Inclination e\n', ' = 10 . 6 degrees compared ( e = \n', '0 . 2\n', '', ' = Stars = \n', '', 'Another article.\n']
        articles = list(diverse.wiki_articles(rows))
        self.assertEqual([r[0] for r in articles], ['Planet', 'Stars'])
        self.assertIn('10 . 6 degrees', articles[0][3])

    def test_ebook_body_excludes_generated_catalog_summary_and_license(self):
        raw = 'Automatically generated catalog summary.\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nHuman prose.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicense.'
        a, b = diverse.book_body(raw)
        self.assertEqual(raw[a:b], 'Human prose.\n')
        with self.assertRaises(ValueError):
            diverse.book_body('Only a catalog summary.')

    def test_author_metadata_must_match_article_beginning(self):
        row = {'metadata': {'author': 'Jane Smith'}, 'text': 'A story with no name present.'}
        self.assertEqual(diverse.byline(row), (None, None))
        row['text'] = 'Jane Smith\nThe story begins here.'
        self.assertEqual(diverse.byline(row)[0], 'Jane Smith')
        self.assertFalse(diverse.named_person('Editorial Team'))

    def test_news_named_credit_does_not_override_translation_disclosure(self):
        row = {'dataset': 'cc_news', 'publisher': 'reuters.com', 'publication_date': '2017-01-01', 'text': 'This was translated using Google Translate. (Reporting by Jane Smith)', 'metadata': {}}
        self.assertEqual(diverse.news_gate(row, self.policy)[0], 'generation_translation_or_extraction_mention_requires_review')

    def test_balancing_limits_publishers_and_does_not_mutate_candidates(self):
        rows = []
        for genre in self.policy['genre_weights']:
            for i in range(120):
                rows.append({'id': f'{genre}/{i}', 'genre': genre, 'text': f'Unique {genre} prose number {i}', 'whitespace_words': 500,
                             'publisher': f'publisher-{i % 7}', 'author': f'Author {i % 11}', 'parent_document_id': f'{genre}/{i}'})
        before = copy.deepcopy(rows)
        selected, allocation = diverse.balanced_select(rows, self.policy)
        target = allocation['capacity_balanced_target_words']
        totals = {}
        for r in selected:
            if r['genre'] == 'journalism': totals[r['publisher']] = totals.get(r['publisher'], 0) + 500
        self.assertTrue(all(v <= target * 0.08 for v in totals.values()))
        self.assertEqual(rows, before)
        self.assertEqual({r['genre'] for r in selected}, set(self.policy['genre_weights']))


if __name__ == '__main__':
    unittest.main()
