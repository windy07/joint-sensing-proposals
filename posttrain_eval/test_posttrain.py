import copy
from pathlib import Path
import unittest
from common import read, digest, write
from design import make_design, choose_best
from paired_statistics import analyze, paired_contrast, world_table


def fixture():
    jobs, records = [], []
    for scenario, count, offset in [('a', 3, 0), ('b', 5, 1000)]:
        for w in range(count):
            for seed in (1, 2, 3):
                for variant, change in [('n0f0', 0), ('n1f0', -2), ('n0f1', -3), ('n1f1', -7)]:
                    job = {'id': f'{scenario}/{w}/{seed}/{variant}', 'scenario': scenario,
                           'world_seed': offset+w, 'arm': 'integrated_agent', 'variant': variant,
                           'model_seed': seed}
                    jobs.append(job)
                    records.append({'job': job, 'status': 'COMPLETE', 'design_digest': 'fixture',
                                    'summary': {'mean_gospa_m': 100 + offset + w + seed + change}})
    return {'jobs': jobs, 'strata': ['a', 'b'], 'model_seeds': [1, 2, 3],
            'design_digest': 'fixture', 'primary_metric': 'mean_gospa_m'}, records


class StatisticalTests(unittest.TestCase):
    def test_paired_worlds_cancel_common_noise(self):
        design, rows = fixture()
        table, missing = world_table(design, rows, 'mean_gospa_m')
        self.assertFalse(missing)
        result = paired_contrast(table, {'integrated_agent/n1f0': 1, 'integrated_agent/n0f0': -1},
                                 ['a', 'b'], draws=1000)
        self.assertEqual(result['paired_worlds'], 8)  # not 24 fits, not 64 repeated frames
        self.assertEqual(result['estimate'], -2)
        self.assertEqual(result['ci95'], [-2, -2])

    def test_equal_scenario_weighting(self):
        table = {('a', i): {'x': 2} for i in range(2)}
        table.update({('b', i): {'x': 10} for i in range(20)})
        result = paired_contrast(table, {'x': 1}, ['a', 'b'], draws=1000)
        self.assertEqual(result['estimate'], 6)
        self.assertEqual(result['ci95'], [6, 6])

    def test_factorial_interaction(self):
        design, rows = fixture()
        report = analyze(design, rows, metrics=['mean_gospa_m'], draws=1000)
        effect = report['metrics']['mean_gospa_m']['estimates']['numeric_feedback_interaction']
        self.assertEqual(effect['estimate'], -2)
        self.assertEqual(effect['ci95'], [-2, -2])

    def test_missing_fit_never_averaged_away(self):
        design, rows = fixture()
        rows[0]['summary']['mean_gospa_m'] = None
        table, missing = world_table(design, rows, 'mean_gospa_m')
        self.assertNotIn('integrated_agent/n0f0', table['a', 0])
        self.assertEqual(missing[0]['valid_fits'], 2)
        self.assertEqual(analyze(design, rows, metrics=['mean_gospa_m'], draws=100)['status'],
                         'PRIMARY_OUTCOMES_INCOMPLETE')

    def test_failed_and_pending_counted(self):
        design, rows = fixture()
        rows[0]['status'] = 'FAILED'
        rows.pop()
        result = analyze(design, rows, metrics=['mean_gospa_m'], draws=100)
        self.assertEqual(result['counts']['FAILED'], 1)
        self.assertEqual(result['counts']['PENDING'], 1)
        self.assertEqual(result['status'], 'INCOMPLETE')

    def test_duplicate_or_wrong_result_rejected(self):
        design, rows = fixture()
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_RESULT'):
            world_table(design, rows + [rows[0]], 'mean_gospa_m')
        rows[0]['design_digest'] = 'different'
        with self.assertRaisesRegex(ValueError, 'DESIGN_MISMATCH'):
            world_table(design, rows, 'mean_gospa_m')

    def test_absent_scenario_no_false_ci(self):
        result = paired_contrast({('a', 1): {'x': 1}, ('a', 2): {'x': 1}}, {'x': 1}, ['a', 'b'], draws=100)
        self.assertIsNone(result['ci95'])
        self.assertIsNone(result['estimate'])

    def test_order_invariance_and_reproducibility(self):
        d, rows = fixture()
        a = analyze(d, rows, metrics=['mean_gospa_m'], draws=101)
        b = analyze(d, list(reversed(rows)), metrics=['mean_gospa_m'], draws=101)
        self.assertEqual(a, b)


class SelectionTests(unittest.TestCase):
    def test_best_is_validation_only_and_earliest_tie(self):
        history = [{'epoch': i, 'optimizer_updates': i*3200, 'total': 1 if i in (3, 9) else 2}
                   for i in range(1, 11)]
        self.assertEqual(choose_best(history)['epoch'], 3)
        with self.assertRaises(ValueError):
            choose_best(history[:-1])
        history[-1]['optimizer_updates'] = 33000
        with self.assertRaises(ValueError):
            choose_best(history)

    def test_manifest_matrix(self):
        original = {'backbones': ['integrated_agent', 'deep_sets', 'set_transformer'],
                    'model_seeds': [633431, 633432, 633433],
                    'world_splits': {'training': [1], 'validation': [2],
                                     'development': list(range(100, 164)),
                                     'confirmation': list(range(1000, 1192))},
                    'scenarios': [{'name': f's{i}', 'overrides': {}} for i in range(8)],
                    'model': {'root_score_budget': 12}, 'source_digest': 'synthetic_test_fixture'}
        design = make_design(original)
        self.assertEqual(sum(j['split'] == 'development' for j in design['jobs']), 1664)
        self.assertEqual(sum(j['split'] == 'confirmation' for j in design['jobs']), 4992)
        self.assertEqual(len(set(j['id'] for j in design['jobs'])), 6656)
        self.assertTrue(all(j['configuration']['root_score_budget'] == 12 for j in design['jobs']))
        self.assertEqual(set(j['arm'] for j in design['jobs'] if j['variant'] == 'n0f1'), {'integrated_agent'})


class ResourceTests(unittest.TestCase):
    def test_counts_training_free_gpus(self):
        from linux_queue import release_count
        gpus = [{'index': i, 'free_gib': 30, 'compute_pids': []} for i in range(3)]
        self.assertEqual(release_count(gpus, {0, 1, 2}, set(), set())[0], 0)
        self.assertEqual(release_count(gpus, {1, 2}, set(), set())[0], 1)
        self.assertEqual(release_count(gpus, {2}, set(), set())[0], 2)
        gpus[1]['free_gib'] = 12
        self.assertEqual(release_count(gpus, {2}, set(), set())[0], 1)

    def test_newly_released_gpu_joins_live_evaluation(self):
        from unittest.mock import patch
        from linux_queue import release_count
        gpus = [{'index': 0, 'free_gib': 20, 'compute_pids': [100]},
                {'index': 1, 'free_gib': 10, 'compute_pids': [101]},
                {'index': 2, 'free_gib': 25, 'compute_pids': [200, 201]}]
        with patch('linux_queue.alive', return_value=True):
            count, available = release_count(gpus, {0, 1}, {200, 201}, {2})
            self.assertEqual(count, 1)
            self.assertEqual([g['index'] for g in available], [2])
            gpus[0].update(free_gib=31, compute_pids=[])
            count, available = release_count(gpus, {1}, {200, 201}, {2})
            self.assertEqual(count, 2)
            self.assertEqual([g['index'] for g in available], [0, 2])
            gpus[0]['compute_pids'] = [999]
            count, available = release_count(gpus, {1}, {200, 201}, {2})
            self.assertEqual(count, 1)
            self.assertEqual([g['index'] for g in available], [2])

    def test_admission_respects_memory_cpu_and_total(self):
        from linux_queue import additions
        hw = {'cpu_quota': 75, 'cpu_used': 10, 'memory_free_gib': 20}
        self.assertEqual(additions(hw, 0, 50), 4)
        hw['memory_free_gib'] = 200
        self.assertEqual(additions(hw, 47, 50), 3)
        hw['cpu_used'] = 70
        self.assertEqual(additions(hw, 0, 50), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
