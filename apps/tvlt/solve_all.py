import os
import json
import copy
import statistics
import time
import random
import pickle

import pytorch_lightning as pl
from torchmetrics import Metric

import torch
import numpy as np
import pprint

from demos import MOSEI_sentiment_model

from dropinf.inference.inference import Inference
from dropinf.profiler.profiler import Profiler
from dropinf.inference.input import *
from dropinf.optimizer.optimizer import Optimizer
from dropinf.profiler.utils import load_profile_result
from dropinf.optimizer.utils import *
from dropinf.utils.cuda import *
from dropinf.utils.file import *
from dropinf.utils.solver import *

from torch.profiler import profile, record_function, ProfilerActivity

# =============== user-defined ===============


def save_b_to_a(batch_to_acc, save_file=None):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "wb") as f:
        pickle.dump(batch_to_acc, f)

def solve_all(save_file=None):
    mod_combo, all_policies_lat = read_batch_metric(
            "./mosei_batch_profiling_result/mosei_batch.json",
            metric_name="latency")

    # all batch sizes
    batch_sizes = np.array([b for b in range(1, 33)])

    # all accuracies
    accuracies = np.arange(0.660017, 0.743196, 0.002)

    # create result matrix
    batch_to_acc = [None] * (1+len(batch_sizes))
    batch_to_acc[0] = [None] * len(accuracies)

    for batch_size in range(1, len(batch_sizes) + 1):
        batch_accs = [None] * len(accuracies)
        for j, acc in enumerate(accuracies):
            batch_accs[j] = list(solve(batch_size=batch_size,
                max_batch_size=32,
                num_mod_policies=3,
                avg_accuracies=[0.7431962, 0.72329473, 0.66001713],
                acc_target=acc,
                metrics=all_policies_lat))
            # process drop policy
            p = batch_accs[j][0]
            policy = {mod_combo[i]: int(p[i]) for i in range(len(mod_combo))}
            batch_accs[j][0] = policy
        batch_to_acc[batch_size] = batch_accs

    # save to file
    to_save = {}
    to_save["batch_list"] = batch_sizes
    to_save["accuracy_list"] = accuracies
    to_save["data"] = batch_to_acc

    save_b_to_a(to_save, save_file=save_file)

if __name__ == '__main__':
    # This is necessary to bypass librosa limitation
    import torch.multiprocessing as mp
    mp.set_start_method("forkserver")
    # mp.set_start_method("spawn")
    import joblib
    import sklearn

    solve_all(save_file="mosei_b_to_a/b_to_a.pkl")
