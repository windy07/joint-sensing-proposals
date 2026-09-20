"""Portable source, feature, label and dependency contracts for this revision.

The source closure is a conservative superset: every local spatial module and
retained graph-agent Python module. Absolute locations are audited separately;
they are not used as portable identities. Historical radial results cannot be
silently reinterpreted as this dataset.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import sys
from pathlib import Path
from .core import ContractError, digest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'retained_campaign' / 'vendor'
from .feature_state import WIDTH,REGISTRY
from .shared_events import EVENT_SPEC
from .actions import START_CONTROL_SPEC
FEATURE_SPEC = {
    'schema': 'ain.spatial.public_joint_numeric_features.stage6.v1', 'width': WIDTH,'state_registry':REGISTRY,'event_scope':EVENT_SPEC,
    'scales': {'position_m': 1000., 'range_bias_m': 1000.,
               'velocity_mps': 50., 'bearing_bias_rad': .05},
    'joint_blocks': 'all unordered block pairs, including zero and diagonal blocks',
    'joint_layout': {'dimensions': [0, 2], 'normalized_covariance_4x4': [2, 18],
                     'correlation_4x4': [18, 34], 'left_mean': [34, 38],
                     'right_mean': [38, 42], 'left_variance': [42, 46],
                     'right_variance': [46, 50], 'endpoint_codes': [88, 90]},
    'endpoint_codes': 'bias=-(index+1); track=id+1; cell=-(101+id); sensor=-(201+id); divided by 1024',
    'identities': 'semantic node/task/owner-issued-track-id, no positional action IDs',
    'action_domain': 'pulse_scan_v4; 8 numeric controls with full-cycle planned start; effective range/angle/frequency joint biases',
    'start_control':START_CONTROL_SPEC,
    'sets_information': 'covariance factors include endpoint codes and marginal descriptors as numeric tokens',
    'float_storage': 'float32 normalized features, float64 posterior remains authoritative',
}
FEATURE_BINDING = digest(FEATURE_SPEC)
LABEL_SPEC = {
    'schema':'ain.stage6.executed_numeric_group_and_return.v1',
    'normalization_m2':10000.,'ranking_floor_m2':100.,
    'ranking_target':'softmax(-(cost-min(cost))/(0.5*max(IQR(cost),100 m2)))',
    'regression':'group log1p(proxy_m2/10000), numeric-conditioned executed-group discounted mean GOSPA^2',
    'loss':'KL target ranking + group proxy + numeric imitation + 0.25 atom + 0.25 state return + 0.25 executed group return',
    'future_return_learning':'gamma=.95 normalized discounted mean; labels offline only; group future head allocates online paid scores',
    'sequence':'semantic node alignment; BPTT across 8-step chunks; detach only at optimizer boundaries; world reset',
    'numeric_labels':'canonical executed controls and masks; same mapping online and offline',
    'feedback_training':'same incoming temporal memory; first encoding alone advances memory',
    'event_scope':EVENT_SPEC,'start_control':START_CONTROL_SPEC,
}

LABEL_BINDING = digest(LABEL_SPEC)


def source_manifest(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else ROOT
    paths = [*sorted((root / 'spatial_lab').rglob('*.py')),
             *sorted((root / 'retained_campaign/vendor/iclr2027_graph_agent').rglob('*.py')),
             *sorted((root.parent/'receiver_revisions').rglob('*.py'))]
    for name in ('stage6.py','stage6_server.py'):
        launcher=root.parents[2]/name
        if launcher.is_file():paths.append(launcher)
    files = {__import__('os').path.relpath(p,root).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(paths)}
    return {'algorithm': 'sha256(canonical_json(relative_posix_path:sha256(bytes)))',
            'files': files, 'digest': digest(files)}


# Freeze the on-disk closure when this interpreter first imports the contract.
# A long-lived process must not claim freshly edited files were its loaded code.
_LOADED_SOURCE_MANIFEST = source_manifest()


def assert_source_unchanged() -> dict:
    current = source_manifest()
    if current != _LOADED_SOURCE_MANIFEST:
        raise ContractError('SOURCE_CHANGED_AFTER_PROCESS_IMPORT')
    return current


def source_digest() -> str:
    return assert_source_unchanged()['digest']


def dependency_contract() -> dict:
    # CPU/CUDA build tags are logged but not mathematical model identities.
    # The public package versions must agree; cross-platform byte hashes need
    # not match. This rule is explicit rather than a silent loader override.
    return {name: importlib.metadata.version(name).split('+', 1)[0]
            for name in ('numpy', 'scipy', 'torch', 'threadpoolctl')}


def bootstrap_vendor() -> None:
    """Refuse an already-loaded foreign homonym instead of evicting modules."""
    expected = (VENDOR / 'iclr2027_graph_agent').resolve()
    for name, module in tuple(sys.modules.items()):
        if name == 'iclr2027_graph_agent' or name.startswith('iclr2027_graph_agent.'):
            path = getattr(module, '__file__', None)
            if path is None or not Path(path).resolve().is_relative_to(expected):
                raise ContractError('FOREIGN_GRAPH_AGENT_IMPORT:' + name + ':' + str(path))
    path = str(VENDOR)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def import_origins() -> dict:
    """Audit actual imports without claiming every file in the closure ran."""
    internal = {}
    external = {}
    for name, module in sorted(tuple(sys.modules.items())):
        path = getattr(module, '__file__', None)
        if not path or not Path(path).is_file():
            continue
        p = Path(path).resolve()
        if name == 'spatial_lab' or name.startswith('spatial_lab.'):
            expected = ROOT / 'spatial_lab'
        elif name == 'iclr2027_graph_agent' or name.startswith('iclr2027_graph_agent.'):
            expected = VENDOR / 'iclr2027_graph_agent'
        elif name == 'receiver_revisions' or name.startswith('receiver_revisions.'):
            expected = ROOT.parent / 'receiver_revisions'
        else:
            if name in ('numpy', 'scipy', 'torch', 'threadpoolctl'):
                external[name] = {'path': str(p),
                    'version': str(getattr(module, '__version__', 'unknown')),
                    'entry_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            continue
        if not p.is_relative_to(expected.resolve()):
            raise ContractError('FOREIGN_LOCAL_IMPORT:' + name + ':' + str(p))
        internal[name] = {'path': str(p), 'relative_path': __import__('os').path.relpath(p,ROOT).replace('\\','/'),
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    return {'internal': internal, 'external_entries': external,
            'external_versions': dependency_contract()}


def policy_configuration(config: dict, method: str) -> dict:
    private={'target_birth_epochs','target_death_epochs','force_delayed_epochs','force_loss_epochs','world_seed','target_templates'}
    return {k:v for k,v in config.items() if k not in private}


def graph_input_binding(view: dict, actions, config: dict) -> str:
    return digest({'view_digest': view['digest'], 'actions': [a.to_dict() for a in actions],
                   'configuration': config, 'feature_binding': FEATURE_BINDING})


def state_digest(state) -> str:
    """Order-independent tensor-state identity (dtype/shape/name plus bytes)."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(digest([name, str(x.dtype), list(x.shape)]).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def verify_checkpoint(ck: dict) -> None:
    if ck.get('schema') != 'ain.spatial.numeric.priority.v6' or ck.get('feature_binding') != FEATURE_BINDING:
        raise ContractError('LEGACY_OR_STALE_SPATIAL_CHECKPOINT')
    if ck.get('label_binding') != LABEL_BINDING:
        raise ContractError('CHECKPOINT_LABEL_BINDING')
    current = assert_source_unchanged()
    saved = ck.get('source_manifest', {})
    if saved != current:
        raise ContractError('CHECKPOINT_SOURCE_CLOSURE')
    if ck.get('dependency_contract') != dependency_contract():
        raise ContractError('CHECKPOINT_DEPENDENCY_VERSIONS')
    if ck.get('model_cfg_digest') != digest(ck.get('cfg')):
        raise ContractError('CHECKPOINT_MODEL_CONFIGURATION')
    if ck.get('state_digest') != state_digest(ck.get('state', {})):
        raise ContractError('CHECKPOINT_STATE_CONTENT')
    if ck.get('dataset_digest') != digest(ck.get('dataset_manifest')):
        raise ContractError('CHECKPOINT_DATASET_BINDING')
    configs = ck.get('allowed_policy_configurations')
    if not configs or any(key != digest(value) for key,value in configs.items()):
        raise ContractError('CHECKPOINT_CONFIGURATION_DOMAIN')
    training = ck.get('training_law_seeds',[])
    development = ck.get('development_law_seeds',[])
    validation = ck.get('validation_law_seeds',[])
    if (not validation or len(set(validation))!=len(validation) or any(type(v)is not int for v in validation) or
        set(validation)&(set(training)|set(development)) or
        set(validation)!={r['world_seed'] for r in ck.get('validation_manifest',[])}):raise ContractError('CHECKPOINT_VALIDATION_PARTITION')
    if (any(type(x) is not int for x in training+development) or
        len(set(training)) != len(training) or len(set(development)) != len(development) or
        set(training) & set(development) or
        set(training) != {row['world_seed'] for row in ck['dataset_manifest']}):
        raise ContractError('CHECKPOINT_WORLD_PARTITION')
    if ck.get('target') != LABEL_SPEC:
        raise ContractError('CHECKPOINT_TARGET_DEFINITION')
    if ck.get('memory_semantics') != 'semantic_world_reset_chunk_bptt_feedback_reread_v6':
        raise ContractError('CHECKPOINT_MEMORY_SEMANTICS')
    bootstrap_vendor()
    import_origins()
