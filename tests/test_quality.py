import json
from pathlib import Path
import unittest

from datasketch import MinHash, MinHashLSH
import quality


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((quality.ROOT / 'config/quality.json').read_text())

    def test_footer_cleanup_preserves_exact_body_and_is_source_specific(self):
        body = ('The reporter explained local policy in detail.\n' * 80).rstrip()
        text = body + '\nYour email address will not be published.'
        end, reason = quality.trim_tail({'text': text, 'publisher': 'newcanadianmedia.ca'}, self.policy, {})
        self.assertEqual(text[:end], body)
        self.assertIsNotNone(reason)
        self.assertEqual(quality.trim_tail({'text': text, 'publisher': 'different.example'}, self.policy, {}), (len(text), None))

    def test_inline_newsletter_or_survey_preserves_following_article_prose(self):
        for publisher, marker in [('milwaukeenns.org', self.policy['footer_lines']['milwaukeenns.org'][0]),
                                  ('freedom.press', self.policy['footer_lines']['freedom.press'][0])]:
            text = 'Detailed article body. ' * 300 + '\n' + marker + '\nThe reporter continued with additional facts and evidence.'
            self.assertEqual(quality.trim_tail({'text': text, 'publisher': publisher}, self.policy, {}), (len(text), None))

    def test_one_ellipsis_or_repeated_human_quote_is_not_related_story_panel(self):
        body = 'A detailed account of the event. ' * 80
        row = {'publisher': 'altnews.in', 'text': body + '\nAn unfinished thought…'}
        frequencies = {('altnews.in', 'An unfinished thought…'): 100}
        self.assertEqual(quality.trim_tail(row, self.policy, frequencies)[0], len(row['text']))
        row['text'] += '\nA second related story…\nA third related story…'
        frequencies.update({('altnews.in', 'A second related story…'): 100, ('altnews.in', 'A third related story…'): 100})
        self.assertEqual(row['text'][:quality.trim_tail(row, self.policy, frequencies)[0]], body.rstrip())

    def test_transitive_similarity_does_not_delete_a_nonduplicate(self):
        adjacency = {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}}
        kept, removed = quality.greedy_keep(['a', 'b', 'c'], adjacency)
        self.assertEqual(kept, {'a', 'c'})
        self.assertEqual(removed, {'b': 'a'})

    def test_retrieval_and_actual_shingles_separate_near_copies_from_shared_footer(self):
        first = ' '.join(f'unique{i}' for i in range(500))
        near = first.replace('unique300', 'changed300')
        unrelated = ' '.join(f'different{i}' for i in range(500))
        a, b, c = [quality.shingles(x + ' Shared newsletter prompt.', 7) for x in [first, near, unrelated]]
        self.assertGreater(quality.jaccard(a, b), 0.92)
        self.assertLess(quality.jaccard(a, c), 0.1)
        lsh = MinHashLSH(num_perm=128, params=(16, 8))
        hashes = []
        template = MinHash(num_perm=128, seed=42, scheme='affine64')
        for terms in [a, b, c]:
            mh = template.copy(); mh.update_batch(t.encode() for t in terms); hashes.append(mh)
        lsh.insert('first', hashes[0])
        self.assertIn('first', lsh.query(hashes[1]))
        self.assertNotIn('first', lsh.query(hashes[2]))


if __name__ == '__main__': unittest.main()
