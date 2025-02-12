import json

import torch

from dropinf.profiler.metrics import *

def set_metrics(recorder):
    r"""
    Set up default metrics for every profiler
    """
    recorder.register_metric(name=f"accuracy", metric=Accuracy())
    recorder.register_metric(name=f"memory", metric=Memory())
    recorder.register_metric(name=f"latency", metric=Latency())

def set_metric(recorder, metric_name=None, metric=None):
    recorder.register_metric(name=metric_name, metric=metric)

def convert_metric(metric, output_format="json"):
    return metric.compute()

def json_serialize(data):
    if torch.is_tensor(data):
        return float(data.cpu().numpy())
    return data

def load_profile_result(result_path=None):
    assert(result_path is not None)
    with open(result_path) as fp:
        data = json.load(fp)
        return data
