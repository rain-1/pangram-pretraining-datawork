import pickle
from pathlib import Path
import tempfile
import unittest

import par3work


class Par3Tests(unittest.TestCase):
    def test_data_only_pickle_loader_refuses_object_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.pkl'
            data = {'book': {'source_paras': ['original'], 'gt_paras': ['English'], 'translator_data': {}}}
            path.write_bytes(pickle.dumps(data))
            self.assertEqual(par3work.load_data_only(path), data)
            class Constructed:
                def __reduce__(self): return (str, ('must not construct',))
            path.write_bytes(pickle.dumps(Constructed()))
            with self.assertRaisesRegex(ValueError, 'Non-data pickle opcode'):
                par3work.load_data_only(path)

    def test_long_aligned_paragraphs_stay_intact_and_missing_human_variants_are_excluded(self):
        config = {'minimum_words': 4, 'maximum_words': 20, 'repo': 'fixture', 'revision': 'fixture'}
        data = {'book_fr': {'source_paras': ['original paragraph one', 'original paragraph two'],
                           'gt_paras': ['four words in translation', 'second four word translation'],
                           'translator_data': {'translator_1': {'translator_paras': ['a human translation here', 'another human translation here']},
                                               'translator_2': {'translator_paras': ['', 'yet another human translation']}}}}
        records, binary, counts = par3work.extract(data, config, 'fixture-sha')
        self.assertEqual(len(records), 5)
        self.assertEqual(counts['missing_human_variants'], 1)
        self.assertEqual(len({r['split'] for r in records}), 1)
        self.assertTrue(all('\n' not in r['text'] for r in records))
        self.assertEqual({r['engine'] for r in binary if r['binary_target'] == 1}, {'Google Translate'})
        self.assertTrue(all(r['book_id'] == 'book_fr' for r in records))
        self.assertTrue(all(r['intended_use'] == 'translation_detector_research_only' for r in records))

    def test_distinct_volumes_of_same_work_share_a_partition(self):
        config = {'minimum_words': 4, 'maximum_words': 20, 'repo': 'fixture', 'revision': 'fixture',
                  'work_group_patterns': {'^war_and_peace_[123]_ru$': 'war_and_peace_ru'}}
        data = {}
        for part in [1, 2, 3]:
            data[f'war_and_peace_{part}_ru'] = {'source_paras': [f'distinct original part {part}'],
                'gt_paras': [f'machine English paragraph part {part}'],
                'translator_data': {'translator_1': {'translator_paras': [f'human English paragraph part {part}']}}}
        records, _, _ = par3work.extract(data, config, 'fixture-sha')
        self.assertEqual(len({r['split_group'] for r in records}), 1)
        self.assertEqual(len({r['split'] for r in records}), 1)

    def test_cluster_metrics_resample_whole_groups_and_require_complete_scores(self):
        rows = [{'id':'h1','split_group':'first','binary_target':0}, {'id':'h2','split_group':'second','binary_target':0},
                {'id':'m1','split_group':'first','binary_target':1}, {'id':'m2','split_group':'second','binary_target':1}]
        scores = [{'id':'h1','score':0.9}, {'id':'h2','score':0.1}, {'id':'m1','score':0.9}, {'id':'m2','score':0.9}]
        report = par3work.cluster_metrics(rows, scores, 0.5, repetitions=100)
        self.assertEqual(report['work_groups'], 2)
        self.assertEqual(report['human_fpr']['value'], 0.5)
        self.assertEqual(report['machine_recall']['cluster_percentile95'], [1.0,1.0])
        with self.assertRaises(ValueError): par3work.cluster_metrics(rows, scores[:1], 0.5)


if __name__ == '__main__': unittest.main()
