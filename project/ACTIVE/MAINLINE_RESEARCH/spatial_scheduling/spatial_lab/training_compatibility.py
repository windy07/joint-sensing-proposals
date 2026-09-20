"""Explicit acceptance of the timing-repair producer for representation-only repair.

The saved producer identity is retained, including on resumed checkpoints.
This gate does not accept earlier feature/label revisions or evaluation cases.
"""
import hashlib,json
from .core import ContractError,digest
from .binding import ROOT,FEATURE_BINDING,LABEL_BINDING,source_manifest

CERTIFICATE_SHA256='43a6f9b0aacdc3028ff18d7ec9b9f44ffc8e779ff8c933baa8e255b5c0fc6678'
PARENT_DIGEST='797989257f80d647a763a305ccdc096e436b7c32052ec886445c106b4b2baa83'

def certificate():
    raw=(ROOT.parents[2]/'STAGE5_BINDINGS/TRAINING_IO_COMPATIBILITY.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CERTIFICATE_SHA256:
        raise ContractError('TRAINING_IO_CERTIFICATE_CHANGED')
    c=json.loads(raw);old=c['parent_source_manifest'];current=source_manifest()
    changed=sorted(n for n in set(old['files'])|set(current['files']) if old['files'].get(n)!=current['files'].get(n))
    if (old['digest']!=PARENT_DIGEST or digest(old['files'])!=PARENT_DIGEST or
        changed!=c['changed_source_files'] or c['feature_binding']!=FEATURE_BINDING or c['label_binding']!=LABEL_BINDING):
        raise ContractError('TRAINING_IO_COMPATIBILITY_SCOPE')
    return c

def accept_parent(saved):
    return saved==certificate()['parent_source_manifest']

def accept_export(source):
    if source==source_manifest()['digest']:return
    if source!=certificate()['parent_source_manifest']['digest']:
        raise ContractError('TRAINING_EXPORT_PRODUCER_NOT_ACCEPTED')
