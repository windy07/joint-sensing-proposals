"""Publication table data with meaningful physical columns and explicit missingness.

Creates tidy CSV and a booktabs LaTeX main table. No synthetic result values.
"""
from pathlib import Path
import argparse
import csv
from common import read, write

MAIN = ['fixed_analytic', 'numeric_optimizer', 'integrated_agent/n0f0', 'integrated_agent/n1f1',
        'deep_sets/n0f0', 'deep_sets/n1f1', 'set_transformer/n0f0', 'set_transformer/n1f1']
NAMES = {'fixed_analytic': 'Analytic fixed controls', 'numeric_optimizer': 'Numerical search',
         'integrated_agent': 'Relational encoder', 'deep_sets': 'Deep Sets style',
         'set_transformer': 'Set self-attention'}
PERFORMANCE = ['mean_gospa_m', 'position_rmse_m', 'matched_target_time_fraction', 'mean_false_tracks']
RESOURCES = ['rf_energy_J', 'receiver_electrical_energy_J', 'transmit_on_s',
             'exposure_energy_density_proxy_J_m2']


def label(name):
    if '/' not in name:
        return NAMES.get(name, name)
    arm, variant = name.split('/')
    suffix = {'n0f0': 'fixed', 'n1f1': 'generated + feedback', 'n1f0': 'generated', 'n0f1': 'fixed + feedback'}[variant]
    return NAMES.get(arm, arm) + ' / ' + suffix


def export(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    estimates = []
    scenarios = []
    for metric, body in report['metrics'].items():
        for name, row in body['estimates'].items():
            ci = row.get('ci95') or [None, None]
            estimates.append({'quantity': name, 'metric': metric, 'unit': body['unit'], 'estimate': row.get('estimate'),
                              'ci95_lower': ci[0], 'ci95_upper': ci[1], 'paired_worlds': row.get('paired_worlds'),
                              'result_status': row['status'], 'collection_status': report['status']})
            for scenario, part in row['per_scenario'].items():
                scenarios.append({'quantity': name, 'metric': metric, 'scenario': scenario,
                                  'estimate': part['mean_difference'], 'paired_worlds': part['paired_worlds'],
                                  'collection_status': report['status']})
    for filename, rows in [('estimates_and_contrasts.csv', estimates), ('by_scenario.csv', scenarios)]:
        with (output / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    def cell(name, metric):
        body = report['metrics'].get(metric, {})
        row = body.get('estimates', {}).get(name, {})
        value = row.get('estimate')
        if value is None:
            return '--'
        scale = 100 if metric == 'matched_target_time_fraction' else 1
        low, high = row['ci95']
        return f'{scale*value:.2f} [{scale*low:.2f}, {scale*high:.2f}]'
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Method & GOSPA (m)$\downarrow$ & Matched RMSE (m)$\downarrow$ & Coverage (\%)$\uparrow$ & False tracks$\downarrow$ \\',
             r'\midrule']
    for name in MAIN:
        lines.append(label(name).replace('_', r'\_') + ' & ' + ' & '.join(cell(name, m) for m in PERFORMANCE) + r' \\')
        if name in ('numeric_optimizer', 'integrated_agent/n1f1', 'deep_sets/n1f1'):
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'main_performance.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(output / 'TABLE_SPECIFICATION.json', {
        'collection_status': report['status'], 'scope': report.get('scope', 'confirmation'),
        'main_performance': PERFORMANCE, 'resource_panel': RESOURCES,
        'scenario_panel': 'eight scenarios, GOSPA and target-time coverage; same methods and worlds',
        'mechanism_panel': 'four inference configurations and paired interaction; three trained seeds',
        'appendix': ['per-fit results', 'conditional sample counts', 'unconfirmed targets', 'complete action timelines'],
        'not_primary_metrics': ['training loss', 'normalized numeric MAE', 'candidate surrogate regret',
                                'candidate count', 'co-scheduled wall time'],
        'runtime_comparison': 'separate matched-load measurement on Linux; co-scheduled logs retained as operational evidence',
        'caption': 'Equal scenario weights; intervals resample paired worlds with trained fits held fixed. '
                   'Matched RMSE uses confirmed matched target times. Coverage and missing-world counts accompany it. '
                   'Absent estimates are shown as dashes. Three fits are not three independent worlds.'})
    return {'status': 'EXPORTED', 'estimate_rows': len(estimates), 'scenario_rows': len(scenarios)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    print(export(read(args.report), args.output))
