"""Fail-closed, portable per-case identity. No simulator or optimizer is called here."""
from __future__ import annotations
from pathlib import Path
import hashlib
from .core import ContractError, digest
from .binding import source_manifest, dependency_contract, FEATURE_BINDING, LABEL_BINDING

CASE_SCHEMA = 'ain.spatial.numeric.case.v4'
RESULT_SCHEMA = 'ain.spatial.numeric.result.v4'

def checkpoint_identity(path):
    if path is None:
        return {'kind':'none', 'sha256':None}
    p=Path(path)
    if not p.is_file():
        raise ContractError('CHECKPOINT_NOT_FOUND:'+str(p))
    return {'kind':'file','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def case_identity(seed, method, config, checkpoint=None):
    if type(seed) is not int or not isinstance(method,str) or not method:
        raise ContractError('INVALID_CASE_IDENTITY')
    body={'schema':CASE_SCHEMA,'result_schema':RESULT_SCHEMA,'world_seed':seed,
          'method':method,'configuration':config,'source':source_manifest(),
          'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,
          'dependencies':dependency_contract(), 'checkpoint':checkpoint or checkpoint_identity(None)}
    return {**body,'digest':digest(body)}

def seal_result(result, identity):
    body={**result,'schema':RESULT_SCHEMA,'case_identity':identity}
    body.pop('content_digest',None)
    return {**body,'content_digest':digest(body)}

def validate_cached(result, expected):
    """The sole reuse gate. Missing, unknown, corrupted and partial records fail."""
    if result.get('schema') != RESULT_SCHEMA:
        raise ContractError('CACHE_UNKNOWN_RESULT_SCHEMA')
    saved=result.get('case_identity')
    if not isinstance(saved,dict) or saved.get('schema')!=CASE_SCHEMA:
        raise ContractError('CACHE_MISSING_OR_UNKNOWN_CASE_BINDING')
    if saved.get('digest')!=digest({k:v for k,v in saved.items() if k!='digest'}):
        raise ContractError('CACHE_IDENTITY_CORRUPT')
    if saved!=expected:
        mismatched=[k for k in set(saved)|set(expected) if saved.get(k)!=expected.get(k)]
        raise ContractError('CACHE_CASE_MISMATCH:'+','.join(sorted(mismatched)))
    if result.get('status')!='COMPLETE':
        raise ContractError('CACHE_NOT_COMPLETE')
    if result.get('content_digest')!=digest({k:v for k,v in result.items() if k!='content_digest'}):
        raise ContractError('CACHE_CONTENT_CORRUPT')
    if (result.get('seed')!=expected['world_seed'] or result.get('method')!=expected['method'] or
        result.get('configuration')!=expected['configuration']):
        raise ContractError('CACHE_PAYLOAD_IDENTITY_MISMATCH')
    if len(result.get('trace',[])) != expected['configuration']['cycles']:
        raise ContractError('CACHE_INCOMPLETE_TRACE')
    return result
