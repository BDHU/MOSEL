import time

from queue import PriorityQueue
from multiprocessing.managers import SyncManager, NamespaceProxy

class EDFQueue(PriorityQueue):
    def get_jobs(self):
        return self.queue

class PriorityQueueManager(SyncManager):
    pass

def EDFQueueManager():
    m = PriorityQueueManager()
    m.start()
    return m

class EDFProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__getattr__', '__setattr__', 'get_jobs')

    def get_jobs(self):
        callmethod = NamespaceProxy.__getattribute__(self, '_callmethod')
        return callmethod('get_jobs')


PriorityQueueManager.register("EDFQueue", EDFQueue, EDFProxy)
# PriorityQueueManager.register("get_q", lambda: q)

# def detect_delay(queue):
#     start_time = time.time_ns() / 1000000
#     for i, j in enumerate(queue):
#         if start_time + j.comp_time - 5 > j.ddl:
#             return queue[:i+1]
#         start_time += j.comp_time

#     return queue
