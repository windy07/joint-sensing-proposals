"""Tidy actual-action and per-cycle tables, with stable trace locations."""
from pathlib import Path
import csv
import json


def export_trace(result, directory, job):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    actions, cycles = [], []
    for frame in result['trace']:
        epoch, commit = frame['epoch'], frame['commit']
        identity = {k: job[k] for k in ('id', 'world_seed', 'scenario', 'arm', 'model_seed', 'variant')}
        identity['epoch'] = epoch
        selected = commit['selected']
        if not selected:
            actions.append({**identity, 'action_index': -1, 'kind': 'idle', 'scan': '[]'})
        for index, action in enumerate(selected):
            actions.append({**identity, 'action_index': index, **{k: v for k, v in action.items() if k != 'scan'},
                            'scan': json.dumps(action.get('scan', []), separators=(',', ':'))})
        meter = commit['meter']
        scores = [r['proxy_m2'] for r in commit['archive']]
        matched = next((r['proxy_m2'] for r in commit['archive'] if r['schedule'] == selected), None)
        cycles.append({**identity, **{k: v for k, v in frame['metric'].items()
                                     if isinstance(v, (int, float, bool, str)) or v is None},
                       'best_paid_proxy_m2': min(scores) if scores else None,
                       'selected_paid_proxy_m2': matched,
                       'feedback_applied': commit.get('feedback_object') is not None,
                       'score_failures': meter['score_failures'],
                       'projection_attempts': len(commit['proposals']),
                       'projection_rejections': sum(p.get('status') == 'REJECTED' for p in commit['proposals']),
                       'remaining_RF_energy_J': frame['remaining_RF_energy_J'],
                       'remaining_transmit_time_s': frame['remaining_transmit_time_s'],
                       'trace_location': f"trace[{epoch}]"})
    for name, rows in [('actions.csv', actions), ('cycles.csv', cycles)]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (directory / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return {'action_rows': len(actions), 'cycle_rows': len(cycles),
            'selection': 'all cycles; no best/worst outcome selection'}
