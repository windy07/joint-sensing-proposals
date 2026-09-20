"""Finite job matrix and completed-epoch checkpoint selection."""
from pathlib import Path
import math
from common import read, write, digest, sha, adapter_digest

ARMS = ('integrated_agent', 'deep_sets', 'set_transformer')
SEEDS = (633431, 633432, 633433)


def variants(arm):
    return ('n0f0', 'n1f1', 'n1f0', 'n0f1') if arm == 'integrated_agent' else ('n0f0', 'n1f1')


def series(job):
    return job['variant'] if job.get('arm') is None else job['arm'] + '/' + job['variant']


def make_design(original):
    if set(original['backbones']) != set(ARMS) or tuple(original['model_seeds']) != SEEDS:
        raise ValueError('UNEXPECTED_MODEL_MATRIX')
    partitions = [set(v) for v in original['world_splits'].values()]
    if any(len(set(v)) != len(v) for v in original['world_splits'].values()):
        raise ValueError('DUPLICATE_WORLD')
    if any(partitions[i] & partitions[j] for i in range(len(partitions)) for j in range(i)):
        raise ValueError('WORLD_PARTITION_OVERLAP')
    jobs = []
    for split in ('development', 'confirmation'):
        for i, seed in enumerate(original['world_splits'][split]):
            regime = original['scenarios'][i % len(original['scenarios'])]
            cfg = {**original['model'], **regime['overrides'], 'scenario': regime['name']}
            choices = [(None, None, v) for v in ('fixed_analytic', 'numeric_optimizer')]
            choices += [(a, s, v) for a in ARMS for s in SEEDS for v in variants(a)]
            for arm, model_seed, variant in choices:
                job = {'split': split, 'world_seed': seed, 'scenario': regime['name'],
                       'arm': arm, 'model_seed': model_seed, 'variant': variant,
                       'configuration': cfg}
                job['id'] = digest(job)[:24]
                jobs.append(job)
    return {'schema': 'radar.posttrain.design.v1', 'status': 'DEVELOPMENT_DESIGN',
            'base_plan_digest': digest(original), 'base_source_digest': original['source_digest'],
            'adapter_digest': adapter_digest(), 'checkpoint_rule': 'minimum validation total over epochs 1..10; earliest tie',
            'stop_updates': 32000, 'original_lr_schedule_epochs': 20,
            'primary_metric': 'mean_gospa_m', 'strata': [s['name'] for s in original['scenarios']],
            'model_seeds': list(SEEDS), 'bootstrap_unit': 'world within scenario; all methods and fitted seeds stay paired',
            'interval_scope': 'world sampling variability for the three retained fitted models',
            'jobs': jobs, 'confirmation_status': 'PLANNED_NOT_FROZEN_NOT_EXECUTABLE'}


def choose_best(history):
    if sorted(r.get('epoch') for r in history) != list(range(1, 11)):
        raise ValueError('NEED_EXACTLY_TEN_VALIDATED_EPOCHS')
    for r in history:
        if r.get('optimizer_updates') != r['epoch'] * 3200 or not math.isfinite(r['total']):
            raise ValueError('INVALID_VALIDATION_HISTORY')
    return min(history, key=lambda r: (r['total'], r['epoch']))


def audit_model(directory, arm, seed, original):
    """Fully verify the chosen best and fixed-epoch files without optimizer work."""
    import torch
    from spatial_lab.binding import verify_checkpoint
    directory = Path(directory)
    receipt = read(directory / 'TRAIN_RECEIPT.json')
    if receipt.get('status') != 'COMPLETE' or receipt.get('optimizer_updates') != 32000:
        raise ValueError('TRAINING_NOT_COMPLETE_AT_TEN')
    final = torch.load(directory / 'epoch_010.pt', map_location='cpu', weights_only=True)
    selected = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    for ck in (final, selected):
        verify_checkpoint(ck)
        if ck['arm'] != arm or ck['model_seed'] != seed or not ck['trained']:
            raise ValueError('CHECKPOINT_IDENTITY')
        if ck['training_purpose'] != 'registered_server_training':
            raise ValueError('CHECKPOINT_PURPOSE')
        if ck['training_configuration'] != original['training']:
            raise ValueError('TRAINING_CONFIGURATION_CHANGED')
        for ck_key, plan_key in [('training_law_seeds', 'training'), ('validation_law_seeds', 'validation'),
                                 ('development_law_seeds', 'development')]:
            if set(ck[ck_key]) != set(original['world_splits'][plan_key]):
                raise ValueError('CHECKPOINT_DATA_PARTITION')
        if not all(torch.isfinite(t).all().item() for t in ck['state'].values()):
            raise ValueError('NONFINITE_MODEL')
    best = choose_best(final['validation_history'])
    if final['optimizer_updates'] != 32000 or selected['optimizer_updates'] != best['optimizer_updates']:
        raise ValueError('CHECKPOINT_SELECTION_MISMATCH')
    if selected['best_validation'] != {k: best[k] for k in ('epoch', 'optimizer_updates', 'total')}:
        raise ValueError('BEST_VALIDATION_MISMATCH')
    if selected['validation_history'] != final['validation_history'][:best['epoch']]:
        raise ValueError('VALIDATION_HISTORY_PREFIX_MISMATCH')
    return {'status': 'VERIFIED', 'arm': arm, 'model_seed': seed,
            'checkpoint': str((directory / 'best.pt').resolve()), 'checkpoint_sha256': sha(directory / 'best.pt'),
            'epoch10_sha256': sha(directory / 'epoch_010.pt'), 'chosen_epoch': best['epoch'],
            'chosen_updates': best['optimizer_updates'], 'validation_total': best['total'],
            'actual_optimizer_updates': 0}


def audit_all(models_root, original):
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            try:
                rows.append(audit_model(Path(models_root) / arm / str(seed), arm, seed, original))
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                rows.append({'arm': arm, 'model_seed': seed, 'status': 'NOT_READY', 'reason': repr(exc)})
    return {'status': 'READY' if all(r['status'] == 'VERIFIED' for r in rows) else 'INCOMPLETE',
            'models': rows, 'actual_optimizer_updates': 0, 'actual_world_advances': 0}
