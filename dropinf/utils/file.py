import json
import os

def read_from_json(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)
        return data

def read_batch_metric(filename, metric_name=None):
    data = read_from_json(filename)
    mod_combo = list(data.keys())
    print(mod_combo)
    all_policy_metrics = []
    for policy, metrics in data.items():
        policy_metrics = {}
        for batch_size, metric in metrics.items():
            batch_size = int(batch_size)
            policy_metrics[batch_size] = metric[metric_name]
        policy_metrics = dict(sorted(policy_metrics.items(),
            key=lambda item: item[0]))
        all_policy_metrics.append(list(policy_metrics.values()))

    return mod_combo, all_policy_metrics
 

