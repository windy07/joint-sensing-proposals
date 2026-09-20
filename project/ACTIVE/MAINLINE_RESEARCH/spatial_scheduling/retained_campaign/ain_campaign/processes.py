"""Small bounded local process supervisor. No SSH, services or credentials."""
from __future__ import annotations
from pathlib import Path
import os,sys,time,signal,subprocess,ctypes
from .common import atomic_json

class WindowsJob:
 def __init__(self,proc):
  from ctypes import wintypes as W
  k=ctypes.WinDLL('kernel32',use_last_error=True);self.k=k
  class BASIC(ctypes.Structure):
   _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
  class IO(ctypes.Structure):_fields_=[(n,ctypes.c_uint64) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
  class EXT(ctypes.Structure):_fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
  k.CreateJobObjectW.restype=W.HANDLE;k.CreateJobObjectW.argtypes=[ctypes.c_void_p,W.LPCWSTR]
  k.SetInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD];k.SetInformationJobObject.restype=W.BOOL
  k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];k.AssignProcessToJobObject.restype=W.BOOL
  k.TerminateJobObject.argtypes=[W.HANDLE,W.UINT];k.CloseHandle.argtypes=[W.HANDLE]
  self.h=k.CreateJobObjectW(None,None)
  if not self.h:raise OSError(ctypes.get_last_error(),'CreateJobObject')
  lim=EXT();lim.BasicLimitInformation.LimitFlags=0x2000 # KILL_ON_JOB_CLOSE
  if not k.SetInformationJobObject(self.h,9,ctypes.byref(lim),ctypes.sizeof(lim)) or not k.AssignProcessToJobObject(self.h,W.HANDLE(int(proc._handle))):
   err=ctypes.get_last_error();self.close();raise OSError(err,'JobObject assignment; refusing unsupervised child')
 def terminate(self):
  if self.h:self.k.TerminateJobObject(self.h,124)
 def close(self):
  if getattr(self,'h',None):self.k.CloseHandle(self.h);self.h=None

def kill_tree(proc,job=None):
 if os.name=='nt':
  if job:job.terminate()
  else:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=False)
 else:
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:pass
 try:proc.wait(timeout=3)
 except subprocess.TimeoutExpired:
  if os.name!='nt':
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  else:proc.kill()
  proc.wait(timeout=3)

def run_child(command,log_path,timeout_s,env_extra=None,cwd=None,cancel_event=None):
 p=Path(log_path);p.parent.mkdir(parents=True,exist_ok=True);env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',**(env_extra or {})};start=time.perf_counter();job=None;proc=None;status='FAILED';code=1;error=None
 with p.open('wb') as log:
  try:
   if timeout_s<=0:raise TimeoutError('CAMPAIGN_DEADLINE')
   kw={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=cwd,**kw)
   if os.name=='nt':job=WindowsJob(proc)
   while True:
    if cancel_event is not None and cancel_event.is_set():kill_tree(proc,job);status='CANCELLED';code=130;break
    left=timeout_s-(time.perf_counter()-start)
    if left<=0:kill_tree(proc,job);status='TIMED_OUT';code=124;break
    try:code=proc.wait(timeout=min(.25,left));status='COMPLETE' if code==0 else 'FAILED';break
    except subprocess.TimeoutExpired:pass
  except BaseException as e:
   error=repr(e)
   if proc is not None and proc.poll() is None:kill_tree(proc,job)
   code=124 if isinstance(e,TimeoutError) else 1
   if isinstance(e,TimeoutError):status='TIMED_OUT'
   if isinstance(e,KeyboardInterrupt):status='INTERRUPTED'
  finally:
   if job:job.close()
   elif proc is not None and os.name!='nt':
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
 result={'command':list(command),'exit_status':code,'status':status,'seconds':time.perf_counter()-start,'pid':None if proc is None else proc.pid,'error':error,'environment_overrides':env_extra or {},'log':str(p),'process_tree_policy':'Windows KILL_ON_JOB_CLOSE; timeout terminate job' if os.name=='nt' else 'isolated POSIX session; SIGTERM then SIGKILL'}
 atomic_json(p.with_suffix('.receipt.json'),result);return result
