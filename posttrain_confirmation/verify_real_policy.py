"""Small real-checkpoint decision checks; zero optimizer/world updates."""
from pathlib import Path
import argparse
import copy
import json
from common import read, write, load_science, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    load_science(args.project)
    import torch
    torch.set_num_threads(1)
    from spatial_lab.spatial_graph import Priority
    from spatial_lab.binding import policy_configuration, source_digest
    from spatial_lab.core import digest
    from spatial_lab.scheduling import run_policy as registered
    from policy import run_policy
    case = read(args.case)
    cfg = policy_configuration(case['configuration'], 'numeric_network')
    seed = case['seed']
    view = case['trace'][0]['causal_view']
    original_view = copy.deepcopy(view)
    policy = Priority(args.checkpoint, device='cpu', evaluation_split='development')
    commits, memories = {}, {}
    for numeric in (False, True):
        for feedback in (False, True):
            key = f'n{int(numeric)}f{int(feedback)}'
            policy.begin_world(seed, cfg)
            method = 'numeric_network' if numeric else 'fixed_network'
            _, commit = run_policy(view, cfg, method, seed, policy,
                                   numeric_parameters=numeric, feedback_enabled=feedback)
            assert view == original_view
            assert not commit['failures'], commit['failures']
            assert commit['meter']['score_calls'] <= cfg['root_score_budget']
            assert commit['meter']['construction'] <= cfg['construction_cap']
            assert len(policy.memory.audit) == 1
            assert policy.memory.last_epoch == 0
            assert (commit['feedback_object'] is not None) == feedback
            assert commit['meter']['network_forwards'] == 1 + int(feedback)
            if not numeric:
                allowed = {digest(r['executed']) for r in commit['proposals'] if r['status'] == 'ACCEPTED'}
                assert all(digest(r['schedule']) in allowed for r in commit['archive'])
                assert commit['meter']['numeric_materializations'] == 0
            commits[key] = commit
            memories[key] = policy.memory.snapshot()
            if numeric == feedback:
                policy.begin_world(seed, cfg)
                _, control = registered(view, cfg, method, seed, policy)
                for field in ('selected', 'archive', 'proposals', 'output', 'feedback_object', 'coverage_guard_status'):
                    assert commit[field] == control[field], (key, field)
    # Feedback rereads the incoming memory; the outgoing first-pass memory is identical.
    for n in (0, 1):
        a, b = memories[f'n{n}f0'], memories[f'n{n}f1']
        assert a['audit'] == b['audit'] and len(a['entries']) == len(b['entries'])
        assert all(x['node'] == y['node'] and torch.equal(x['state'], y['state'])
                   for x, y in zip(a['entries'], b['entries']))
    assert commits['n0f0']['proposals'] == commits['n0f1']['proposals']
    for f in (0, 1):
        assert commits[f'n1f{f}']['meter']['numeric_materializations'] > 0
    receipt = {'status': 'PASSED', 'checkpoint_sha256': sha(args.checkpoint), 'source_digest': source_digest(),
               'case_seed': seed, 'tested_epoch': 0, 'cells': {
                   k: {'score_calls': c['meter']['score_calls'], 'construction': c['meter']['construction'],
                       'network_forwards': c['meter']['network_forwards'],
                       'feedback_applied': c['feedback_object'] is not None,
                       'numeric_materializations': c['meter']['numeric_materializations']}
                   for k, c in commits.items()},
               'registered_n0f0_n1f1_parity': True, 'fixed_parameters_preserved': True,
               'single_memory_advance': True, 'actual_world_advances': 0, 'actual_optimizer_updates': 0}
    write(args.output, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
