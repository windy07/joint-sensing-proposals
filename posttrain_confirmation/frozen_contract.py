"""Validation of the prespecified confirmation design and selected weights."""
from common import ROOT, read, digest


def validate_frozen(design, contract=None):
    contract = read(ROOT/'EXPERIMENT_CONTRACT.json') if contract is None else contract
    expected = 'sha256:' + digest({k:v for k,v in contract.items() if k != 'freeze_digest'})
    if contract.get('freeze_digest') != expected or design.get('contract_digest') != expected:
        raise ValueError('CONFIRMATION_CONTRACT_CHANGED')
    if contract.get('mode') != 'confirmatory' or contract.get('launch_gate', {}).get('status') != 'GO':
        raise ValueError('CONFIRMATION_NOT_READY')
    if design.get('confirmation_status') != 'FROZEN' or not contract['embargo']['confirmation_unopened']:
        raise ValueError('CONFIRMATION_NOT_FROZEN')
    if any(v['status'] not in ('PASS', 'NOT_APPLICABLE') for v in contract['admissibility'].values()):
        raise ValueError('CONFIRMATION_READINESS_CHANGED')
    bindings = contract['bindings']
    if (digest(design['jobs']) != bindings['jobs_digest'] or
        design['adapter_digest'] != bindings['adapter_digest'] or
        design['base_source_digest'] != bindings['science_digest']):
        raise ValueError('CONFIRMATION_BINDING_CHANGED')
    if any(j['split'] != 'confirmation' for j in design['jobs']):
        raise ValueError('CONFIRMATION_SPLIT_CHANGED')
    if set(j['world_seed'] for j in design['jobs']) != set(contract['sample_size_plan']['world_seeds']):
        raise ValueError('CONFIRMATION_WORLDS_CHANGED')
    return contract


def validate_checkpoint(contract, arm, seed, actual):
    key = arm + '/' + str(seed)
    if contract['bindings']['checkpoints'].get(key) != actual:
        raise ValueError('FROZEN_CHECKPOINT_CHANGED:' + key)
