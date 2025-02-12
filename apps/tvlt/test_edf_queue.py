import os
import json
import copy
import statistics
import time
import random
import pickle
import argparse

import pytorch_lightning as pl
from torchmetrics import Metric
from multiprocessing import Manager

import torch
import numpy as np
import pprint

from demos import MOSEI_sentiment_model
from config import *

from dropinf.inference.inference import Inference
from dropinf.profiler.profiler import Profiler
from dropinf.inference.input import *
from dropinf.optimizer.optimizer import Optimizer
from dropinf.profiler.utils import load_profile_result
from dropinf.optimizer.utils import *
from dropinf.utils.cuda import *
from dropinf.utils.file import *
from dropinf.utils.solver import *
from dropinf.inference.queue import *
from dropinf.inference.job import *
from dropinf.inference.database import *
from dropinf.inference.policy import *
from dropinf.inference.monitor import *
from dropinf.inference.worker import worker
from dropinf.inference.trace import *

from torch.profiler import profile, record_function, ProfilerActivity

# =============== user-defined ===============


class MyInfer(Inference):
    
    def forward(self, model, batch, args):
        """
        TODO: implement your own forward function
        # mdoel: torch model
        # batch: must best a dictionary
        E.g. {"video_data": tensor[], "audio_data": tensor[]}
        """

        encoder_last_hidden_outputs, *_ = model(video=batch["video_data"], audio=batch["audio_data"])
        sentiment_score = model.classifier(encoder_last_hidden_outputs).squeeze().data
        return sentiment_score

class MyInputManager(InputManager):

    def split_batch(self, batch, split_sections=None):
        """
        TODO: Define how to split a given batch. By default, the function takes
        two arugments:

        batch: dictionary of data
            e.g. {"audio": torch.tensor, "video": torch.tensor, "index": ...}
        split_sections: array of splitted batch sizes
            e.g [12, 14, 6]

        return an array of splitted batches

        Can also remove other unnecessary data in a batch
        """
        if split_sections is None:
            return batch

        # remove unnecessary data in batch
        keys_to_remove = [k for k in batch.keys() if k != "video_data"
                and k != "audio_data"]
        for k in keys_to_remove:
            batch.pop(k, None)

        # remove split size with 0
        split_sections = [v for v in split_sections if v > 0]


        assert(sum(split_sections) == batch["audio_data"].size(dim=0))
        splitted_video = torch.split(batch["video_data"], split_sections)
        splitted_audio = torch.split(batch["audio_data"], split_sections)
        
        splitted_batch = [batch.copy() for i in range(len(split_sections))]

        for i, b in enumerate(splitted_batch):
            for k, _ in b.items():
                if k == "video_data":
                    b[k] = splitted_video[i]
                elif k == "audio_data":
                    b[k] = splitted_audio[i]

        return splitted_batch

    def take_batch(self, batch, size=None):
        """
        TODO: this function just takes the specified size of a batch
        batch: a dict of tensors
        """
        if size is None:
            return batch

        new_batch = {"video_data": None, "audio_data": None}

        splitted_video = torch.clone(batch["video_data"][:size])
        splitted_audio = torch.clone(batch["audio_data"][:size])
        new_batch["video_data"] = splitted_video
        new_batch["audio_data"] = splitted_audio
        return new_batch
        
def rand_gen(qps):
    rand_nums = []
    rand_sum = 0

    while rand_sum < qps:
        # num = random.randint(1, qps - rand_sum) # numpy norm
        num = int(np.random.normal(loc=1, scale=6))
        if num <= 0:
            num = 1
        rand_nums.append(num)
        rand_sum += num

    if rand_sum > qps:
        rand_nums[-1] -= (rand_sum - qps)
    
    return rand_nums

def get_time(num_request, profiler_result):
    return profiler_result["None"][str(num_request)]["latency"]

def interval_gen(num_requests):
    return [1 / sum(num_requests) * i for i in num_requests]

def def_acc_gen(min, max):
    return random.uniform(min, max)

def producer(min_acc, max_acc, queue, recv_queue, batch, input_manager, remove_mod,
        database, profiler_filename, tracefile, tracer):

    profiler_result = read_from_json(profiler_filename)
    f = open(tracefile, "r")

    # os.makedirs(os.path.dirname("trace_results/submitted_jobs.txt"), exist_ok=True)
    # result = open("trace_results/submitted_jobs.txt", "a")
    lines = f.readlines()
    j_id = 0
    acc_target = def_acc_gen(min_acc, max_acc)

    # warmup
    for i in range(4):
        deadline = time.time_ns()/1000000 + 4000
        j = Job(
                ddl=deadline,
                job_id= i - 100,
                acc_target=acc_target,
                payload=batch,
                num_requests=32,
                start_time=time.time_ns() / 1000000,
                sec_range=-1,
                database=database)
        queue.put(j)
        time.sleep(2)


    for i, line in enumerate(lines):
        l = line.split()
        sec = int(l[0])
        qps = int(l[1])

        num_requests = rand_gen(qps)
        intervals = interval_gen(num_requests)

        for interval, num_request in zip(intervals, num_requests):
            estimate_lat = get_time(num_request, profiler_result)
            b = input_manager.take_batch(batch, size=num_request)

            deadline = time.time_ns()/1000000 + estimate_lat + 55
            # deadline = time.time_ns()/1000000 + estimate_lat + 17
            j = Job(
                    ddl=deadline,
                    job_id=j_id,
                    acc_target=acc_target,
                    payload=b,
                    num_requests=num_request,
                    start_time=time.time_ns() / 1000000,
                    sec_range=sec,
                    database=database)
            queue.put(j)
            time.sleep(interval-0.003)
            j_id += 1

            # if j.job_id > 34:
            with tracer.lock:
                tracer.submitted_jobs.value += 1
                tracer.submitted_requests.value += j.batch_size
                tracer.submitted_latency.append(estimate_lat)
                tracer.submitted_accuracy.append(j.acc_target)
                tracer.current_sub_job_id.value = j.job_id

        # tracer write # of jobs submitted
        num_jobs = len(num_requests)
        print("num_jobs: ", num_jobs)
        line_to_w = str(sec) + " " + str(num_jobs)
    #     result.write(line_to_w+"\n")
    
    
    # result.close() 

    # get all responses
    response_id = 0
    # for i in range(num_jobs):
    while True:
        r = recv_queue.get()
        print("respond id: ", r)
        if r == "end":
            return

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace_file', '-t', required=True, type=str,
                    dest='trace_file',
                    help='The name of the trace file using twitter trace')
    ap.add_argument('--twitter_trace_file', required=True, type=str,
                    dest='twitter_trace_file',
                    help='the generated twitter trace file')
    ap.add_argument('--policy', required=True, type=str,
                    dest='policy',
                    help='queue optimization policy')
    ap.add_argument('--profiled_results', required=True, type=str,
                    dest='profiled_results',
                    help='The .json file generated during offline stage')
    ap.add_argument('--pkl', required=True, type=str,
                    dest='pkl',
                    help='The .pkl file')
    ap.add_argument('--min_acc', required=True, type=float,
                    dest='min_acc',
                    help='The minimum accuracy')
    ap.add_argument('--max_acc', required=True, type=float,
                    dest='max_acc',
                    help='The maximum accuracy')
    return ap.parse_args()

if __name__ == '__main__':
    # This is necessary to bypass librosa limitation
    import torch.multiprocessing as mp
    mp.set_start_method("forkserver")
    import joblib
    import sklearn
    from model.modules import Transformer
    from model.data.datamodules.multitask_datamodule import MTDataModule
    import random
    random.seed(0)
    args = get_args()


    ####################################################################
    # TODO get the model and a batch
    ####################################################################

    # context setup
    _config = get_mosei_config()
    pl.seed_everything(_config["seed"])

    # dataset setup
    dm = MTDataModule(_config, dist=False)
    dm.setup("test")
    dm.prepare_data()

    # device = 2
    device = 'cpu'
    quantize = False
    arguments = None
    exp_name = f'{_config["exp_name"]}'

    batch = next(iter(dm.test_dataloader()))

    model = MOSEI_sentiment_model(device=device)

    ####################################################################
    # end TODO
    ####################################################################

    # infer setup
    mosei_infer = MyInfer()
    mosei_input_manager = MyInputManager(
        modality_names = {
            "video": ["video_data"],
            "audio": ["audio_data"]
        }
    )


    # read from profiled results
    database = None
    with open(args.pkl, "rb") as f:
        b_to_a = pickle.load(f)
        database = Database(batch_list=b_to_a["batch_list"],
                accuracies=b_to_a["accuracy_list"], data=b_to_a["data"])
        print(database.data[1])


    # send queue
    # edf_m = EDFQueueManager()
    # edf_queue = edf_m.PriorityQueue()
    # edf_queue = edf_m.EDFQueue()

    # recv queue
    send_queue = mp.Queue()
    watcher_send_queue = mp.Queue()
    recv_queue = mp.Queue()
    work_queue_overhead = mp.Value('d', 0.0)
    work_queue_lock = mp.Lock()
    last_job_start_time = mp.Value('d', 0.0)
    last_job_lock = mp.Lock()

    m = Manager()
    tracer_lock = mp.Lock()
    submitted_latency = m.list()
    submitted_accuracy = m.list()
    processed_latency = m.list()
    processed_accuracy = m.list()
    # tracer_file = "./trace_results/aggressive_results.txt"
    # tracer_file = "./trace_results/random_results.txt"
    # tracer_file = "./trace_results/fair_results.txt"
    # tracer_file = "./trace_results/no_policy_results.txt"
    
    tracer_file = args.trace_file
    tracer = Tracer(lock = tracer_lock,
                    submitted_latency=submitted_latency,
                    submitted_accuracy=submitted_accuracy,
                    processed_latency=processed_latency,
                    processed_accuracy=processed_accuracy,
                    file_name = tracer_file)

    # twitter_trace = "../dropinf_data_proc/twitter_04_24_norm.txt"
    twitter_trace = args.twitter_trace_file
    # profiler_name = "./profiled_results.json"
    profiler_name = args.profiled_results

    # start worker
    worker_process = mp.Process(target = worker,
            args = (
                model,
                mosei_infer,
                watcher_send_queue,
                recv_queue,
                mosei_input_manager,
                True,
                work_queue_overhead,
                work_queue_lock,
                last_job_start_time,
                last_job_lock,
                tracer,
                quantize,
                device,
                arguments))
    worker_process.start()


    # start watcher
    watcher_func = None
    if args.policy == "no_policy":
        watcher_func = no_policy
    elif args.policy == "maximize":
        watcher_func = fairness_policy
    elif args.policy == "random":
        watcher_func = rand_policy
    elif args.policy == "aggressive":
        watcher_func = fit_acc
    else:
        raise Exception("Unknown policy specified")
    
    watcher = mp.Process(target=watcher_func,
            args = (send_queue,
                watcher_send_queue,
                database,
                work_queue_overhead,
                work_queue_lock,
                last_job_start_time,
                last_job_lock))
    watcher.start()

    time.sleep(5)

    # start producer
    producer_processes = []
    j_id= 0
    num_procs = 1
    for i in range(num_procs):
        deadline = random.randint(3, 9)
        producer_process = mp.Process(target = producer,
                args = (
                    args.min_acc,
                    args.max_acc,
                    send_queue,
                    recv_queue,
                    batch,
                    mosei_input_manager,
                    False,
                    database,
                    profiler_name,
                    twitter_trace,
                    tracer))
        producer_processes.append(producer_process)
        j_id += 1

    start = time.time()
    for p in producer_processes:
        p.start()

    # start trace proc
    trace_process = mp.Process(target=trace_proc,
                            args = (tracer,))
    trace_process.start()

    trace_process.join()
    active = mp.active_children()
    for child in active:
        child.terminate()
    exit()

    for p in producer_processes:
        p.join()
    end = time.time()
    print("latency: ", end - start)

    worker_process.join()
    watcher.join()