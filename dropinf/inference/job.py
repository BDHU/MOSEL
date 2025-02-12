from dataclasses import dataclass, field
from typing import Any

import multiprocessing

import torch

from dropinf.inference.database import *

@dataclass(order=True)
class EDFJob:
    """
    Earliest-deadline-first inference job
    """
    priority: float
    item: Any=field(compare=False)

class Job:

    def __init__(
            self,
            comp_time=None,
            ddl=None,
            job_id=None,
            acc_target=None,
            acc_in_use=None,
            drop_policy=None,
            payload=None,
            num_requests=None,
            start_time=None,
            start_exec_time=None,
            sec_range=None,
            database=None,
            debug=None):

        if comp_time is None:
            p = database.get_max_acc_by_b(batch_size=num_requests)
            if database.get_acc(p) < acc_target:
                raise ValueError("accuracy target too high")
            self.comp_time = database.get_latency(p)
        else:
            self.comp_time = comp_time
        self.ddl = ddl
        self.job_id = job_id
        self.acc_target = acc_target
        if acc_in_use is None:
            p = database.get_max_acc_by_b(batch_size=num_requests)
            if database.get_acc(p) < acc_target:
                raise ValueError("accuracy target too high")
            self.acc_in_use = database.get_acc(p)
        else:
            self.acc_in_use = acc_in_use

        if drop_policy is None:
            p = database.get_max_acc_by_b(batch_size=num_requests)
            if database.get_acc(p) < acc_target:
                raise ValueError("accuracy target too high")
            self.drop_policy = database.get_mod_policy(p)
        else:
            self.drop_policy = {k: v for k, v in drop_policy.items() if v > 0}
        self.payload = payload

        self.modified = False
        self.batch_size = num_requests

        self.start_time = start_time
        self.start_exec_time = start_exec_time
        self.sec_range = sec_range
        self.debug = debug

    def update(
            self,
            comp_time=None,
            drop_policy=None,
            modified=None,
            acc_in_use=None):
        self.comp_time = comp_time
        self.drop_policy = {k: v for k, v in drop_policy.items() if v > 0}
        self.modified = modified
        self.acc_in_use = acc_in_use
        

    def __eq__(self, other):
        return ((self.ddl, self.job_id) ==
                (other.ddl, other.job_id))

    def __lt__(self, other):
        return ((self.ddl, self.job_id) <
                (other.ddl, other.job_id))


def fill_job(job, database, start_time, time_budget=None):
    b_index, acc_index = database.get_index(acc=job.acc_in_use,
            batch_size=job.batch_size)

    metrics = database.get_b_metrics(batch_size=job.batch_size) 
    new_time_budget = time_budget
    old_lat = job.comp_time
    old_policy = job.drop_policy

    if time_budget > 0:
        j = acc_index
        n = len(metrics)
        while j < n:
            candidate = metrics[j]
            candidate_end_time = start_time + database.get_latency(candidate)
            if candidate_end_time >= job.ddl or new_time_budget <= 0:
                return job, new_time_budget

            job.update(comp_time=database.get_latency(candidate),
                        drop_policy=database.get_mod_policy(candidate),
                        modified=(True if old_policy != \
                                database.get_mod_policy(candidate) else False),
                        acc_in_use=database.get_acc(candidate))
            extra_time = job.comp_time - old_lat
            new_time_budget = time_budget - extra_time

            j += 1

        return job, new_time_budget
    elif time_budget < 0:
        j = acc_index
        while j >= 0:
            candidate = metrics[j]
            candidate_acc = database.get_acc(candidate)
            candidate_end_time = start_time + database.get_latency(candidate)
            if candidate_acc < job.acc_target:
                return job, new_time_budget

            job.update(comp_time=database.get_latency(candidate),
                        drop_policy=database.get_mod_policy(candidate),
                        modified=(True if old_policy != \
                                database.get_mod_policy(candidate) else False),
                        acc_in_use=database.get_acc(candidate))
            saved_time = old_lat - job.comp_time
            new_time_budget = time_budget + saved_time
            # print("time budget: ", time_budget)
            # print(candidate)
            # print("old lat: ", old_lat)
            # print("job comp time: ", job.comp_time)
            # print("saved_time: ", saved_time)
            # print("new time budget: ", new_time_budget)
            # print("id: ", job.job_id)
            # print("\n\n")
            j -= 1

        return job, new_time_budget
