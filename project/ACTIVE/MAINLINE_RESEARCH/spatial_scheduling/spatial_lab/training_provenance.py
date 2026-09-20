"""Explicit raw-teacher import across reference-feature and time-label repairs.

Original cases are validated and retained as producer records. Only their
causal graph and physical control labels are rebuilt by the consumer. This gate is
never used to reuse an evaluation case or a trained checkpoint.
"""
from pathlib import Path
import hashlib, json
from .core import ContractError, digest
from .binding import ROOT, source_manifest, FEATURE_BINDING, LABEL_BINDING, LABEL_SPEC
from .case_identity import validate_cached

CERTIFICATE_SHA256 = '23ef18bbe6661c3bf74ffdd1ee5682a23e2867eab5902e7ef31adaeda389c406'
CERTIFICATE_PATH = ROOT.parents[2]/'STAGE5_BINDINGS/ACTION_TIMING_PRODUCER.json'
CHANGED_FILES = frozenset(['retained_campaign/vendor/iclr2027_graph_agent/memory.py', 'spatial_lab/actions.py', 'spatial_lab/binding.py', 'spatial_lab/core.py', 'spatial_lab/feature_state.py', 'spatial_lab/sequence_training.py', 'spatial_lab/server_pipeline.py', 'spatial_lab/stage4_tests.py', 'spatial_lab/training_cache.py', 'spatial_lab/training_compatibility.py', 'spatial_lab/training_provenance.py'])


def producer_certificate():
    raw = CERTIFICATE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CERTIFICATE_SHA256:
        raise ContractError('TEACHER_COMPATIBILITY_CERTIFICATE_CHANGED')
    cert = json.loads(raw)
    old = cert['source_manifest']
    if old['digest'] != digest(old['files']):
        raise ContractError('TEACHER_PRODUCER_MANIFEST_CORRUPT')
    current = source_manifest()
    changed = {name for name in set(old['files'])|set(current['files'])
               if old['files'].get(name) != current['files'].get(name)}
    if changed != CHANGED_FILES:
        raise ContractError('TEACHER_COMPATIBILITY_SOURCE_SCOPE_CHANGED')
    if (digest(cert['producer_label_spec'])!=cert['label_binding'] or
        cert['consumer_label_spec']!=LABEL_SPEC or cert['consumer_label_binding']!=LABEL_BINDING or
        cert['consumer_feature_binding']!=FEATURE_BINDING):
        raise ContractError('TEACHER_LABEL_REENCODING_BINDING')
    changed_labels={k for k in set(cert['producer_label_spec'])|set(LABEL_SPEC)
                    if cert['producer_label_spec'].get(k)!=LABEL_SPEC.get(k)}
    if changed_labels!={'numeric_labels','start_control'}:
        raise ContractError('TEACHER_LABEL_REENCODING_SCOPE')
    return cert


def validate_training_case(result, expected):
    if result.get('case_identity') == expected:
        validate_cached(result, expected)
        return None
    cert = producer_certificate()
    if (expected['method'] not in cert['teacher_policies'] or
        expected['checkpoint'] != {'kind':'none','sha256':None} or
        result.get('fixture_only') or
        result.get('evaluation_split') not in ('training','development-diagnostic')):
        raise ContractError('TEACHER_IMPORT_SCOPE')
    producer = {**expected, 'source':cert['source_manifest'],
                'feature_binding':cert['feature_binding'],'label_binding':cert['label_binding']}
    producer['digest'] = digest({k:v for k,v in producer.items() if k!='digest'})
    validate_cached(result, producer)
    if (result.get('source_digest') != producer['source']['digest'] or
        result.get('source_manifest') != producer['source'] or
        result.get('feature_binding') != producer['feature_binding']):
        raise ContractError('TEACHER_PRODUCER_PAYLOAD_BINDING')
    for frame in result['trace']:
        commit = frame['commit']
        if (commit.get('feature_binding') != producer['feature_binding'] or
            commit.get('label_binding') != producer['label_binding'] or
            commit.get('method') != expected['method'] or
            commit.get('capacity_audit',{}).get('network_present') is not False or
            commit.get('digest') != digest({k:v for k,v in commit.items() if k!='digest'})):
            raise ContractError('TEACHER_COMMIT_PRODUCER_BINDING')
    return {'schema':'ain.stage5.teacher.consumer_provenance.v1',
            'producer_source_digest':producer['source']['digest'],
            'producer_feature_binding':producer['feature_binding'],
            'producer_label_binding':producer['label_binding'],
            'producer_case_identity':producer['digest'],
            'consumer_source_digest':expected['source']['digest'],
            'consumer_feature_binding':FEATURE_BINDING,
            'consumer_label_binding':LABEL_BINDING,
            'compatibility_certificate_sha256':CERTIFICATE_SHA256,
            'raw_case_rewritten':False,
            'interpretation':'original analytic teacher trajectory and paid scores; rebuilt causal features and full-cycle physical start labels; exact executable teacher reconstruction required'}
