"""Recall metrics use cardinality-first matching and never count duplicates as TP."""
import unittest
from nnunetv2.evaluation.particle_recall import match_particles, evaluate_cases


class RecallTests(unittest.TestCase):
    def test_optional_instance_and_center_audit(self):
        import numpy as np
        from nnunetv2.evaluation.particle_audit import instance_correspondence, center_correspondence
        cc = np.zeros((5, 5, 5), int)
        cc[1, 1, 1:3] = 1
        ids = np.zeros_like(cc)
        ids[1, 1, 1], ids[1, 1, 2] = 8, 20
        stats = instance_correspondence(cc, cc > 0, ids)
        self.assertEqual(stats['gt_id_count'], 2)
        self.assertEqual(stats['cc_with_multiple_ids'], 1)
        cc[1, 1, 2] = 2
        ids[1, 1, 2] = 8
        self.assertEqual(instance_correspondence(cc, cc > 0, ids)['ids_in_multiple_cc'], 1)
        stats = center_correspondence(cc, [[1, 1, 1], [1, 1, 1], [10, 0, 0]])
        self.assertEqual((stats['cc_with_multiple_centers'], stats['centers_outside_volume']), (1, 1))

    def test_matching_cardinality_precedes_distance(self):
        # Greedy nearest matching takes pred0->gt0 and misses gt1; optimal matches both.
        result = match_particles([[0, 0, 0], [2, 0, 0]], [[.5, 0, 0], [-1, 0, 0]], 1.5)
        self.assertEqual(result['tp'], 2)
        self.assertEqual(result['fn'], 0)

    def test_duplicates_and_empty_cases(self):
        result = match_particles([[0, 0, 0]], [[0, 0, 0], [.1, 0, 0], [9, 0, 0]], 1)
        self.assertEqual((result['tp'], result['fp'], result['duplicate_candidates']), (1, 2, 1))
        self.assertEqual(match_particles([], [[0, 0, 0]], 1)['fp'], 1)
        self.assertEqual(match_particles([[0, 0, 0]], [], 1)['fn'], 1)

    def test_macro_rare_class_and_budget_curves(self):
        cases = [dict(case_id='tomo', valid_region_confirmed=True, valid_volume=100,
            classes={'abundant': dict(gt=[[0, 0, 0]], predictions=[dict(center=[0, 0, 0], score=.2)]),
                     'rare': dict(gt=[[5, 5, 5]], predictions=[])})]
        result = evaluate_cases(cases, {'abundant': 1, 'rare': 1})
        self.assertEqual(result['macro_recall'], .5)
        self.assertEqual(evaluate_cases(cases, {'abundant': 1, 'rare': 1}, threshold=.5)['macro_recall'], 0)
        self.assertEqual(evaluate_cases(cases, {'abundant': 1, 'rare': 1}, budget=0)['macro_recall'], 0)
