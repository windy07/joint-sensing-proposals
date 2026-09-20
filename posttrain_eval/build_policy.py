"""Development-only generator of the explicit, reviewable policy adapter.

Run once after reviewing a new registered source. Never used by evaluation.
"""
from pathlib import Path
import argparse
import hashlib


def build(project, destination):
    path = Path(project) / 'ACTIVE/MAINLINE_RESEARCH/spatial_scheduling/spatial_lab/scheduling.py'
    source = path.read_text(encoding='utf-8')
    start = source.index('def _run_policy(')
    end = source.index('\n\n\ndef run_policy(', start)
    text = source[start:end]
    substitutions = [
        ('def _run_policy(view,config,method,seed,learned=None):',
         'def _factorial_policy(view,config,method,seed,learned,numeric_parameters,feedback_enabled):'),
        ("numeric=method in ('numeric_optimizer','numeric_network','numeric_random_priority')",
         'numeric=bool(numeric_parameters)'),
        ('if numeric and not feedback_done and meter.score_calls>=',
         'if feedback_enabled and not feedback_done and meter.score_calls>='),
        ("            queue=gen.challenge(best,feedback_object)+refined+queue;feedback_done=True",
         "            if numeric:\n                queue=gen.challenge(best,feedback_object)+refined+queue\n"
         "            else:\n                # Frozen projected actions: only the unscored order changes.\n"
         "                queue=rank([a for a in queue if a not in seen])\n"
         "                gen.feedback.append({'feedback':feedback_object,'mode':'fixed_pool_reranking',\n"
         "                                     'continuous_parameters_changed':False})\n"
         "            feedback_done=True"),
    ]
    for before, after in substitutions:
        if text.count(before) != 1:
            raise ValueError('SOURCE_LAYOUT_CHANGED:' + before)
        text = text.replace(before, after)
    # Only feedback proposals are conditional; the feedback encoding always runs.
    before = "                for proposal in learned.proposals(config['max_simultaneous'],meter=meter):\n"
    if text.count(before) != 1:
        raise ValueError('FEEDBACK_PROPOSAL_LOCATION_CHANGED')
    text = text.replace(before,
        "                for proposal in (learned.proposals(config['max_simultaneous'],meter=meter) if numeric else []):\n")
    # The adapter lives outside spatial_lab; preserve these imports explicitly.
    text = text.replace('from .feedback import', 'from spatial_lab.feedback import')
    text = text.replace('from .functional_fixture import', 'from spatial_lab.functional_fixture import')
    header = '''"""Independent numeric-generation / feedback intervention for Stage6 evaluation.

The body retains the registered candidate evaluator, resource projection, coverage
rule and physical model. Only the two inference interventions differ. This module
does not change any training source or checkpoint metadata.
"""
from pathlib import Path
import hashlib
from spatial_lab import scheduling as base
from spatial_lab.scheduling import *

EXPECTED_SCHEDULING_SHA = %r


def check_base():
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != EXPECTED_SCHEDULING_SHA:
        raise ContractError('POSTTRAIN_SCHEDULING_SOURCE_CHANGED')


''' % hashlib.sha256(path.read_bytes()).hexdigest()
    footer = '''

def run_policy(view, config, method, seed, learned=None, *, numeric_parameters, feedback_enabled):
    check_base()
    if method not in ('fixed_network', 'numeric_network') or learned is None:
        raise ContractError('FACTORIAL_REQUIRES_LEARNED_POLICY')
    with decision_linear_work() as work:
        selected, commit = _factorial_policy(view, config, method, seed, learned,
                                             numeric_parameters, feedback_enabled)
        legal(selected, view, config)
    commit['meter']['numpy_linear_algebra_calls'] = work
    commit['meter']['joint_decoder_calls'] = learned.decode_calls
    commit['meter']['joint_decoder_seconds'] = learned.decoder_seconds
    commit['intervention'] = {
        'numeric_parameters': bool(numeric_parameters), 'feedback_enabled': bool(feedback_enabled),
        'fixed_feedback_semantics': 'rerank remaining fixed projected schedules; no new numeric actions',
        'feedback_applied': commit['feedback_object'] is not None,
        'temporal_memory_updates_per_cycle': 1,
    }
    if not numeric_parameters:
        if commit['meter']['numeric_materializations'] or commit['meter']['numeric_optimization_calls']:
            raise ContractError('FIXED_PARAMETERS_CHANGED')
    if commit['meter']['score_calls'] > config['root_score_budget']:
        raise ContractError('POSTTRAIN_SCORE_BUDGET')
    if commit['meter']['construction'] > config['construction_cap']:
        raise ContractError('POSTTRAIN_CONSTRUCTION_BUDGET')
    commit['digest'] = digest({k: v for k, v in commit.items() if k != 'digest'})
    return selected, commit
'''
    Path(destination).write_text(header + text + footer, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--out', default=str(Path(__file__).with_name('policy.py')))
    args = ap.parse_args()
    build(args.project, args.out)
