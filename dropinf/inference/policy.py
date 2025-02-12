import time
import func_timeout

from dropinf.inference.job import *
from dropinf.utils.solver import queue_solve

def aggressive_drop(
        violator_id,
        jobs, 
        work_queue_overhead, 
        start_time,
        database):

    work_queue_overhead = work_queue_overhead - \
            (time.time_ns() / 1000000 - start_time)
    start_time += work_queue_overhead
    i = 0
    # n = len(jobs)
    n = violator_id + 1
    while i < n:
        j = jobs[i]
        _, x = database.get(acc_target=j.acc_target,
                batch_size=j.batch_size)

        if x is None:   # can't meet acc target
            jobs.pop(i)
            n -= 1
            continue

        new_lat = database.get_latency(x)
        new_policy = database.get_mod_policy(x)
        new_acc_in_use = database.get_acc(x)
        j.update(comp_time=new_lat,
                drop_policy=new_policy,
                modified=True,
                acc_in_use=new_acc_in_use)

        if start_time + j.comp_time - 5 > j.ddl:
            print("dropped by aggressive")
            print("job id: ", j.job_id)
            print("job ddl: ", j.ddl)
            print("start time: ", start_time)
            print("j comp time: ", j.comp_time)
            print("estimated finish time: ", start_time + j.comp_time)
            print("\n\n")
            jobs.pop(i)
            n -= 1
            continue

        start_time += j.comp_time
        i += 1

def random_fill(
        violator_id,
        jobs,
        work_queue_overhead,
        start_time,
        diff,
        database):
    # goal: get diff to 0 as close as possible
    if violator_id == len(jobs):
        return

    work_queue_overhead = work_queue_overhead - \
            (time.time_ns() / 1000000 - start_time)

    start_time += work_queue_overhead
    i = 0
    n = violator_id + 1
    diff_is_negative = True if diff <= 0 else False
    diff_is_positive = True if diff > 0 else False
    while i < n:
        j = jobs[i]
        jobs[i], diff = fill_job(j, database, start_time, time_budget=diff)
        if diff > 0 and diff_is_negative is True:
            return
        elif diff <= 0 and diff_is_positive is True:
            raise ValueError("optimization shouldn't deplete time budget")
        # TODO: update start time
        start_time += jobs[i].comp_time
        i += 1
    # if we reached here, we need to drop the job
    jobs.pop(violator_id)


def fair_accuracy(
        violator_id,
        jobs,
        work_queue_overhead,
        start_time,
        diff,
        database):
    if violator_id == len(jobs):
        return
    # executed = time.time_ns() / 1000000 - start_time
    # work_queue_overhead = work_queue_overhead - \
    #         (time.time_ns() / 1000000 - start_time)
    # total_time_budget_ 
    # total_time_budget += diff



    i = 0
    n = violator_id + 1
    diff_is_negative = True if diff <= 0 else False
    diff_is_positive = True if diff > 0 else False

    affected_jobs = jobs[:violator_id + 1] 
    
    # 0. calc current queue total estimated latency 
    total_time_budget = 0
    for j in affected_jobs:
        total_time_budget += j.comp_time
    total_time_budget += diff
    total_time_budget *= 0.91
   
    # 1. for each job, build the latency matrix 
    all_accuracies = []
    all_latencies = []
    for i, job in enumerate(affected_jobs): 
        # get_job_policies() 
        low_id, _ = database.get(acc_target=job.acc_target,
                batch_size=job.batch_size)
        policies = database[job.batch_size][low_id:]
        
        accuracies = [database.get_acc(p) for p in policies]
        latencies = [database.get_latency(p) for p in policies]
        assert(len(accuracies) == len(latencies))
        if len(accuracies) < 4: # to satisfy gekko
            last_acc = accuracies[-1]
            last_lat = latencies[-1]
            length_diff = 4 - len(accuracies)
            for k in range(length_diff):
                accuracies.append(last_acc)
                latencies.append(last_lat)

        all_accuracies.append(accuracies)
        all_latencies.append(latencies)
    assert(len(all_accuracies) == len(all_latencies))

    index_decisions = None
    solver_lat = None
    try:
        max_wait_time = 0.1
        # index_decisions, _, _, _, _, solver_lat = queue_solve(all_latencies,
        #         all_accuracies,
        #         total_time_budget)
        index_decisions, _, _, _, _, solver_lat = func_timeout.func_timeout(
                max_wait_time, queue_solve, args=[all_latencies,
                    all_accuracies,
                    total_time_budget]
                )
    except:
        random_fill(violator_id, jobs,
                work_queue_overhead - 1000 * max_wait_time,
                start_time, diff, database)
        return

    # update policy
    assert(len(index_decisions) == (1 + violator_id))
    for i, decision in enumerate(index_decisions):
        job = jobs[i]
        low_id, _ = database.get(acc_target=job.acc_target,
                batch_size=job.batch_size)
        policies = database[job.batch_size][low_id:]
        
        decision = low_id + decision
        if decision > len(policies) - 1:
            decision = -1

        # new decision
        x = database[job.batch_size][decision]
        new_lat = database.get_latency(x)
        new_policy = database.get_mod_policy(x)
        new_acc_in_use = database.get_acc(x)
        jobs[i].update(comp_time=new_lat,
                drop_policy=new_policy,
                modified=True,
                acc_in_use=new_acc_in_use)
        i += 1