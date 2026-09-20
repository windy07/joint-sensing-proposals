from __future__ import annotations

from dataclasses import asdict,dataclass
from hashlib import sha256
import json,random
from pathlib import Path
from typing import Any,Mapping

try:
 import numpy as np
except Exception: np=None  # type: ignore[assignment]
try:
 import torch
except Exception: torch=None  # type: ignore[assignment]


def file_sha256(path:Path)->str:
 h=sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


@dataclass(frozen=True,slots=True)
class CheckpointMetadata:
 schema_version:str
 epoch:int
 global_step:int
 model_config_sha256:str
 feature_schema_sha256:str
 split_manifest_sha256:str
 training_config_sha256:str
 source_tree_sha256:str
 train_jsonl_sha256:str
 validation_jsonl_sha256:str
 train_joint_jsonl_sha256:str
 validation_joint_jsonl_sha256:str
 validation_metrics:Mapping[str,float]
 checkpoint_selection_split:str="validation"


def save_atomic_checkpoint(path:Path,*,model,optimizer,scheduler,metadata:CheckpointMetadata,extra:Mapping[str,Any]|None=None)->str:
 if torch is None: raise RuntimeError("PyTorch is required to save checkpoints")
 if metadata.checkpoint_selection_split!="validation": raise ValueError("checkpoint selection must use validation data only")
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
 payload={
  'metadata':asdict(metadata),'model_state':model.state_dict(),
  'optimizer_state':None if optimizer is None else optimizer.state_dict(),
  'scheduler_state':None if scheduler is None else scheduler.state_dict(),
   'python_random_state':random.getstate(),
   'numpy_random_state':None if np is None else {
      'bit_generator':str(np.random.get_state()[0]),
      'state':torch.as_tensor(np.random.get_state()[1].astype('int64')),
      'position':int(np.random.get_state()[2]),
      'has_gauss':int(np.random.get_state()[3]),
      'cached_gaussian':float(np.random.get_state()[4]),
   },
  'torch_random_state':torch.random.get_rng_state(),
  'cuda_random_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
  'extra':dict(extra or {}),
 }
 torch.save(payload,tmp); tmp.replace(path)
 digest=file_sha256(path)
 path.with_suffix(path.suffix+'.sha256').write_text(digest+'  '+path.name+'\n',encoding='utf-8')
 return digest


def load_verified_checkpoint(path:Path,*,expected:Mapping[str,str],map_location='cpu',expected_file_sha256:str|None=None):
 if torch is None: raise RuntimeError("PyTorch is required to load checkpoints")
 digest=file_sha256(path)
 sidecar=path.with_suffix(path.suffix+'.sha256')
 if expected_file_sha256 is None and sidecar.is_file():
  expected_file_sha256=sidecar.read_text(encoding='utf-8').strip().split()[0]
 if expected_file_sha256 is not None and digest.lower()!=str(expected_file_sha256).lower():
  raise ValueError(f"checkpoint SHA-256 mismatch: {digest} != {expected_file_sha256}")
 try:
  payload=torch.load(path,map_location=map_location,weights_only=True)
 except TypeError as exc:
  raise RuntimeError("safe checkpoint loading requires a PyTorch version with weights_only support") from exc
 meta=payload.get('metadata',{})
 for key,value in expected.items():
  if str(meta.get(key))!=str(value): raise ValueError(f"checkpoint {key} mismatch: {meta.get(key)!r} != {value!r}")
 return payload,digest
