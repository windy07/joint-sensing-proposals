"""Ordered, one-batch-ahead thread loading of immutable CPU world records.

This module changes transport only. The dataset's validation, world ordering,
batch boundaries and the training loop remain authoritative and unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
import threading,time

class InputStats:
    def __init__(self):
        self.lock=threading.Lock();self.reset()
    def reset(self):
        self.values={'transport':'ordered_threads_one_batch_ahead','worlds_loaded':0,
                     'batches_delivered':0,'pending_worlds':0,'active_reads':0,
                     'max_active_reads':0,'load_wall_sum_s':0.,'consumer_waiting':False,'last_event_ns':time.time_ns()}
    def update(self,**values):
        with self.lock:self.values.update(values,last_event_ns=time.time_ns())
    def event(self,**deltas):
        with self.lock:
            for k,v in deltas.items():self.values[k]=self.values.get(k,0)+v
            self.values['max_active_reads']=max(self.values['max_active_reads'],self.values['active_reads'])
            self.values['last_event_ns']=time.time_ns()
    def snapshot(self):
        with self.lock:return dict(self.values)

class OrderedThreadLoader:
    def __init__(self,dataset,order,batch,workers,stats=None):
        if type(batch)is not int or batch<1 or type(workers)is not int or workers<0:
            raise ValueError('INVALID_INPUT_LOADER_DIMENSIONS')
        self.dataset=dataset;self.order=list(order);self.batch=batch;self.workers=min(4,workers)
        self.stats=stats or InputStats()
    def __len__(self):return (len(self.order)+self.batch-1)//self.batch
    def __iter__(self):
        if not self.order:return
        # Original non-persistent process workers start each iterator with a
        # fresh checked-file set. Preserve that verification frequency.
        checked=getattr(self.dataset,'checked',None)
        if self.workers and isinstance(checked,set):checked.clear()
        def fetch(i):
            started=time.monotonic();self.stats.event(active_reads=1)
            try:
                item=self.dataset[i]
                self.stats.event(worlds_loaded=1)
                return item
            finally:self.stats.event(active_reads=-1,load_wall_sum_s=time.monotonic()-started)
        if self.workers==0:
            for start in range(0,len(self.order),self.batch):
                result=[fetch(i) for i in self.order[start:start+self.batch]]
                self.stats.event(batches_delivered=1)
                yield result
            return
        pool=ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='radar_input')
        pending=[]
        def submit(start):
            indexes=self.order[start:start+self.batch]
            self.stats.update(pending_worlds=len(indexes))
            return [pool.submit(fetch,i) for i in indexes]
        try:
            pending=submit(0)
            for start in range(0,len(self.order),self.batch):
                # Read results in sampler order even when IO completes out of order.
                self.stats.update(consumer_waiting=True)
                result=[future.result() for future in pending]
                self.stats.update(consumer_waiting=False)
                pending=submit(start+self.batch) if start+self.batch<len(self.order) else []
                if not pending:self.stats.update(pending_worlds=0)
                self.stats.event(batches_delivered=1)
                yield result
                # Do not retain a previous batch while waiting for the next one.
                result=None
        finally:
            for future in pending:future.cancel()
            pool.shutdown(wait=True,cancel_futures=True)
            self.stats.update(pending_worlds=0,consumer_waiting=False)

def loader(dataset,order,batch,workers,stats=None):
    return OrderedThreadLoader(dataset,order,batch,workers,stats)
