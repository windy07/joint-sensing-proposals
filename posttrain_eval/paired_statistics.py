"""Scenario-stratified, paired-world estimates for a fixed set of trained fits.

No model seed, target, or time step is counted as a new independent world.
Missing conditional outcomes remain missing; all missing denominators are saved.
"""
from collections import defaultdict
import math
import numpy as np
from design import series

METRICS = {
    'mean_gospa_m': ('m', 'lower', 'world mean GOSPA; p=2, alpha=2, cutoff from configuration'),
    'mean_localization_sum_m2': ('m2', 'lower', 'world mean localization contribution to squared GOSPA'),
    'position_rmse_m': ('m', 'lower', 'mean of world RMSEs conditional on matched confirmed target times'),
    'position_error_p95_m': ('m', 'lower', 'mean of within-world 95th percentiles on matched target times'),
    'velocity_rmse_mps': ('m/s', 'lower', 'mean of world velocity RMSEs conditional on matched target times'),
    'matched_target_time_fraction': ('1', 'higher', 'matched confirmed target times divided by true target times in each world'),
    'missed_target_time_fraction': ('1', 'lower', 'unmatched true target times divided by true target times'),
    'mean_false_tracks': ('tracks', 'lower', 'world mean false confirmed tracks'),
    'not_confirmed_count': ('targets', 'lower', 'targets not confirmed by the end of the episode'),
    'max_lost_interval_s': ('s', 'lower', 'longest loss interval after a first confirmation; null without confirmation'),
    'id_switches': ('count', 'lower', 'cumulative assignment identity switches per world'),
    'rf_energy_J': ('J', 'lower', 'cumulative emitted RF energy'),
    'receiver_electrical_energy_J': ('J', 'lower', 'cumulative receiver electrical energy'),
    'transmit_on_s': ('s', 'lower', 'cumulative transmitter on-time'),
    'exposure_energy_density_proxy_J_m2': ('J/m2', 'lower', 'cumulative energy density at the specified monitor; not an interception probability'),
    'final_bias_rmse_m': ('m', 'lower', 'final effective range-bias RMSE'),
    'final_frequency_bias_rmse_Hz': ('Hz', 'lower', 'final effective frequency-bias RMSE'),
    'final_angle_bias_rmse_deg': ('degree', 'lower', 'final angular-bias RMSE'),
    'paid_root_evaluations': ('count', 'lower', 'total actual root-candidate evaluations'),
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_records(design, records):
    expected = {j['id']: j for j in design['jobs']}
    if len(expected) != len(design['jobs']):
        raise ValueError('DUPLICATE_DESIGN_JOB')
    observed = {}
    for record in records:
        ident = record['job']['id']
        if ident not in expected or record['job'] != expected[ident]:
            raise ValueError('UNREGISTERED_OR_CHANGED_JOB')
        if ident in observed:
            raise ValueError('DUPLICATE_RESULT:' + ident)
        if record.get('design_digest') != design.get('design_digest'):
            raise ValueError('RESULT_DESIGN_MISMATCH')
        if record.get('status') not in ('COMPLETE', 'FAILED', 'RUNNING'):
            raise ValueError('UNKNOWN_RESULT_STATUS')
        observed[ident] = record
    return expected, observed


def world_table(design, records, metric):
    expected, observed = validate_records(design, records)
    groups = defaultdict(list)
    for ident, job in expected.items():
        groups[(job['scenario'], job['world_seed'], series(job))].append(ident)
    table, missing = {}, []
    for (regime, world, name), ids in sorted(groups.items()):
        values, reasons = [], []
        for ident in ids:
            r = observed.get(ident)
            if r is None:
                reasons.append({'id': ident, 'reason': 'PENDING'})
            elif r['status'] != 'COMPLETE':
                reasons.append({'id': ident, 'reason': r['status']})
            elif not finite(r.get('summary', {}).get(metric)):
                reasons.append({'id': ident, 'reason': 'NULL_OR_NONFINITE_METRIC'})
            else:
                values.append(float(r['summary'][metric]))
        if reasons:
            missing.append({'scenario': regime, 'world_seed': world, 'series': name,
                            'expected_fits': len(ids), 'valid_fits': len(values), 'reasons': reasons})
        else:
            table.setdefault((regime, world), {})[name] = float(np.mean(values))
    return table, missing


def paired_contrast(table, coefficients, strata, *, draws=10000, seed=90231):
    """Coefficients apply to within-world fit means, before any resampling."""
    if draws < 100 or not coefficients or any(not finite(c) for c in coefficients.values()):
        raise ValueError('INVALID_BOOTSTRAP_CONFIGURATION')
    rng = np.random.default_rng(seed)
    per_stratum, all_differences = {}, []
    bootstrap = np.zeros(draws)
    for regime in strata:
        values = np.array([sum(coefficients[n] * row[n] for n in coefficients)
                           for (s, w), row in sorted(table.items())
                           if s == regime and all(n in row for n in coefficients)], dtype=float)
        per_stratum[regime] = {'paired_worlds': len(values),
                              'mean_difference': float(values.mean()) if len(values) else None}
        if len(values):
            all_differences.extend(values.tolist())
            bootstrap += values[rng.integers(len(values), size=(draws, len(values)))].mean(1) / len(strata)
    if any(v['paired_worlds'] < 2 for v in per_stratum.values()):
        return {'status': 'INSUFFICIENT_PAIRED_WORLDS', 'per_scenario': per_stratum,
                'estimate': None, 'ci95': None}
    estimate = float(np.mean([v['mean_difference'] for v in per_stratum.values()]))
    return {'status': 'ESTIMATED', 'estimate': estimate,
            'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'per_scenario': per_stratum, 'paired_worlds': len(all_differences),
            'scenario_weighting': 'equal', 'bootstrap_draws': draws, 'bootstrap_seed': seed,
            'uncertainty': 'paired world sampling within each scenario; fitted models held fixed'}


def contrasts(names):
    result = {n + ' minus numeric_optimizer': {n: 1., 'numeric_optimizer': -1.}
              for n in names if n != 'numeric_optimizer' and 'numeric_optimizer' in names}
    p = 'integrated_agent/'
    if all(p + v in names for v in ('n0f0', 'n1f0', 'n0f1', 'n1f1')):
        result.update({
            'numeric_without_feedback': {p + 'n1f0': 1., p + 'n0f0': -1.},
            'numeric_with_feedback': {p + 'n1f1': 1., p + 'n0f1': -1.},
            'feedback_with_fixed_parameters': {p + 'n0f1': 1., p + 'n0f0': -1.},
            'feedback_with_generated_parameters': {p + 'n1f1': 1., p + 'n1f0': -1.},
            'numeric_feedback_interaction': {p + 'n1f1': 1., p + 'n1f0': -1., p + 'n0f1': -1., p + 'n0f0': 1.},
        })
    return result


def seed_descriptives(design, records, metric):
    """Seed-to-seed dispersion, separate from the world-bootstrap interval."""
    groups = defaultdict(dict)
    for r in records:
        j = r['job']
        if r['status'] == 'COMPLETE' and finite(r.get('summary', {}).get(metric)):
            groups[(series(j), j['model_seed'])][j['world_seed']] = float(r['summary'][metric])
    fits = defaultdict(list)
    for name, seed in sorted(groups, key=str):
        jobs = [j for j in design['jobs'] if series(j) == name and j['model_seed'] == seed]
        expected = {j['world_seed'] for j in jobs}
        row = {'model_seed': seed, 'worlds_expected': len(expected), 'worlds_valid': len(groups[name, seed]), 'mean': None}
        if expected == groups[name, seed].keys():
            row['mean'] = float(np.mean([np.mean([groups[name, seed][j['world_seed']]
                for j in jobs if j['scenario'] == s]) for s in design['strata']]))
        fits[name].append(row)
    return {name: {'fits': rows, 'sample_sd_across_fit_means': float(np.std([r['mean'] for r in rows], ddof=1))
                  if len(rows) == len(design['model_seeds']) and all(r['mean'] is not None for r in rows) else None}
            for name, rows in fits.items()}


def analyze(design, records, *, metrics=None, draws=10000):
    expected, observed = validate_records(design, records)
    counts = {s: sum(r['status'] == s for r in observed.values()) for s in ('COMPLETE', 'FAILED', 'RUNNING')}
    counts['PENDING'] = len(expected) - len(observed)
    names = sorted({series(j) for j in expected.values()})
    output = {'status': 'COMPLETE' if counts['COMPLETE'] == len(expected) else 'INCOMPLETE',
              'expected_episodes': len(expected), 'counts': counts,
              'estimate_basis': 'complete paired worlds for each contrast; missing outcomes are listed per metric',
              'metrics': {}}
    for metric in (metrics or METRICS):
        table, missing = world_table(design, records, metric)
        estimands = {name: {name: 1.} for name in names}
        estimands.update(contrasts(names))
        output['metrics'][metric] = {'unit': METRICS[metric][0], 'better': METRICS[metric][1],
            'definition': METRICS[metric][2], 'missing': missing,
            'estimates': {name: paired_contrast(table, coeff, design['strata'], draws=draws)
                          for name, coeff in estimands.items()},
            'seed_descriptives': seed_descriptives(design, records, metric)}
        if metric == design.get('primary_metric') and missing and output['status'] == 'COMPLETE':
            output['status'] = 'PRIMARY_OUTCOMES_INCOMPLETE'
    return output
