import torch
from dropinf.profiler.utils import *

class Recorder:

    def __init__(self):
        self.metric_names = set()

        # init cuda time recorder
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def timer_start(self):
        self.start.record()

    def timer_end(self):
        self.end.record()
        torch.cuda.synchronize()

    def timer_elapsed(self):
        return self.start.elapsed_time(self.end)

    def register_metric(self, name=None, metric=None):
        setattr(self, name, metric)
        self.metric_names.add(name)

    def to_dict(self):
        result_dict = {}
        for m_n in self.metric_names:
            metric = getattr(self, m_n)
            avg = json_serialize(metric.compute())
            stdev = json_serialize(metric.stdev())
            result_dict[m_n] = (avg, stdev)

        return result_dict
