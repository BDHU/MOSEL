import time
from dropinf.inference.policy import *

def detect_delay(queue, work_queue_overhead, start_time):
    #TODO
    # 1. lock to get actual job start time
    # 2. worker warmup
    estimate_start_time = start_time
    total_overhead = work_queue_overhead
    total_overhead -= (time.time_ns() / 1000000 - start_time)


    estimate_start_time += total_overhead
    difference = 0
    for i, j in enumerate(queue):
        estimate_end_time = estimate_start_time + j.comp_time
        difference = j.ddl - estimate_end_time
        if estimate_end_time > j.ddl:
            return i, difference
        estimate_start_time += j.comp_time

    return len(queue), difference
        
def no_policy(recv_queue, send_queue, database, work_queue_overhead,
        work_queue_lock, last_job_start_time, last_job_lock, interval=0.0003):
    # 1. every interval, look at the queue to see if there is ddl violation 
    jobs_queue = []

    while True:
        num_jobs = 0
        jobs_buffer = []
        while not recv_queue.empty() and num_jobs < 30:
            j = recv_queue.get()
            j.ddl += (time.time_ns() / 1000000 - j.start_time) + 5
            jobs_buffer.append(j)
            num_jobs += 1

        # TODO: calc overhead, put jobs to consumer queue
        wq = 0
        if len(jobs_queue) > 0:
            j = jobs_queue.pop(0)
            with work_queue_lock:
                work_queue_overhead.value += j.comp_time
                wq = work_queue_overhead.value
            send_queue.put(j)
        
        jobs_queue = jobs_queue + jobs_buffer
        jobs_queue.sort(key=lambda x: x.ddl)
        if len(jobs_queue) > 0:
            for j in jobs_queue:
                print(j.job_id)
                print(j.ddl)

        # if violator_id != len(jobs_queue):
        #     # print(violator_id)
        #     # print(len(jobs_queue))
        #     # print("start aggressive drop!!!!!!!!!!")
        #     aggressive_drop(violator_id, jobs_queue, wq_overhead, database)

        # print("overhead: ", e - s)
        interval = max(0.0002, wq/1000 * 0.75)
        time.sleep(interval)



def fit_acc(recv_queue, send_queue, database, work_queue_overhead,
        work_queue_lock, last_job_start_time, last_job_lock, interval=0.0003):
    # 1. every interval, look at the queue to see if there is ddl violation 
    jobs_queue = []

    while True:
        num_jobs = 0
        jobs_buffer = []
        while not recv_queue.empty() and num_jobs < 30:
            j = recv_queue.get()
            j.ddl += (time.time_ns() / 1000000 - j.start_time) + 5
            jobs_buffer.append(j)
            num_jobs += 1

        # TODO: calc overhead, put jobs to consumer queue
        if len(jobs_queue) > 0:
            j = jobs_queue.pop(0)
            with work_queue_lock:
                work_queue_overhead.value += j.comp_time
            send_queue.put(j)
        
        jobs_queue = jobs_queue + jobs_buffer
        jobs_queue.sort(key=lambda x: x.ddl)

        start_time = time.time_ns() / 1000000 + 3
        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value


        wq_overhead = 0
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value


        violator_id, _ = detect_delay(jobs_queue, wq_overhead, start_time)
        if violator_id != len(jobs_queue):
            with last_job_lock:
                if last_job_start_time.value > 0:
                    start_time = last_job_start_time.value

            wq_overhead = 0
            with work_queue_lock:
                wq_overhead = work_queue_overhead.value

            aggressive_drop(violator_id, jobs_queue, wq_overhead, start_time,
                    database)

        # print("overhead: ", e - s)
        # interval = max(0.0025, wq_overhead * 0.9 / 1000)

        with work_queue_lock:
            wq_overhead = work_queue_overhead.value
        start_time = time.time_ns() / 1000000 + 3
        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value
        wait = wq_overhead - (time.time_ns()/1000000 - start_time)
        # interval = max(0.0025, wait * 0.8 / 1000)
        interval = max(0.001, wait * 0.75 / 1000)
        time.sleep(interval)
        # time.sleep(0.0025)

def rand_policy(recv_queue, send_queue, database, work_queue_overhead,
        work_queue_lock, last_job_start_time, last_job_lock, interval=0.0003):
    print("random_policy init---------------:w")
    # 1. every interval, look at the queue to see if there is ddl violation 
    jobs_queue = []

    while True:
        num_jobs = 0
        jobs_buffer = []
        while not recv_queue.empty() and num_jobs < 30:
            j = recv_queue.get()
            j.ddl += (time.time_ns() / 1000000 - j.start_time) + 5
            jobs_buffer.append(j)
            num_jobs += 1

        # TODO: calc overhead, put jobs to consumer queue
        if len(jobs_queue) > 0:
            j = jobs_queue.pop(0)
            with work_queue_lock:
                work_queue_overhead.value += j.comp_time
            send_queue.put(j)
        
        jobs_queue = jobs_queue + jobs_buffer
        jobs_queue.sort(key=lambda x: x.ddl)

        start_time = time.time_ns() / 1000000 + 3
        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value

        wq_overhead = 0
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value

        violator_id, difference = detect_delay(jobs_queue, wq_overhead,
                start_time)

        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value

        wq_overhead = 0
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value

        random_fill(violator_id, jobs_queue, wq_overhead, start_time,
                difference, database)
        # if violator_id != len(jobs_queue):
        #     # print(violator_id)
        #     # print(len(jobs_queue))
        #     # print("start aggressive drop!!!!!!!!!!")
        #     aggressive_drop(violator_id, jobs_queue, wq_overhead, database)

        # print("overhead: ", e - s)
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value
        start_time = time.time_ns() / 1000000 + 3
        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value
        wait = wq_overhead - (time.time_ns()/1000000 - start_time)
        interval = max(0.0001, wait * 0.75 / 1000)
        time.sleep(interval)
        # time.sleep(0.0003)

def fairness_policy(recv_queue, send_queue, database, work_queue_overhead,
        work_queue_lock, last_job_start_time, last_job_lock, interval=0.0003):
    # 1. every interval, look at the queue to see if there is ddl violation 
    jobs_queue = []

    while True:
        num_jobs = 0
        jobs_buffer = []
        while not recv_queue.empty() and num_jobs < 20:
            j = recv_queue.get()
            j.ddl += (time.time_ns() / 1000000 - j.start_time) + 5
            jobs_buffer.append(j)
            num_jobs += 1

        if len(jobs_queue) > 0:
            j = jobs_queue.pop(0)
            with work_queue_lock:
                work_queue_overhead.value += j.comp_time
            send_queue.put(j)
        
        jobs_queue = jobs_queue + jobs_buffer
        jobs_queue.sort(key=lambda x: x.ddl)

        wq_overhead = 0
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value

        # hide solver latency ~= 41ms
        solver_lat = 100
        if wq_overhead <= 0:
            i = 0
            n = len(jobs_queue)
            while i < n:
                pre = jobs_queue[i]
                if pre.comp_time < solver_lat:
                    with work_queue_lock:
                        work_queue_overhead.value += pre.comp_time
                    send_queue.put(pre)
                    jobs_queue.pop(i)
                    n -= 1
                    solver_lat -= pre.comp_time
                else:
                    # find a latency that just suppassed the overhead and submit
                    _, pre_p = database.get(acc_target=pre.acc_target,
                            batch_size=pre.batch_size)
                    new_lat = database.get_latency(pre_p)
                    new_policy = database.get_mod_policy(pre_p)
                    new_acc_in_use = database.get_acc(pre_p)
                    pre.update(comp_time=new_lat,
                                drop_policy=new_policy,
                                modified=True,
                                acc_in_use=new_acc_in_use)
                    with work_queue_lock:
                        work_queue_overhead.value += pre.comp_time
                    send_queue.put(pre)
                    jobs_queue.pop(i)
                    n -= 1
                    break
                i += 1

        start_time = time.time_ns() / 1000000 + 3
        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value

        wq_overhead = 0
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value

        violator_id, difference = detect_delay(jobs_queue, wq_overhead, start_time)

        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value

        wq_overhead = 0
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value

        f_s = time.time_ns() / 1000000
        fair_accuracy(violator_id, jobs_queue, wq_overhead, start_time,
                difference, database)
        f_e = time.time_ns() / 1000000
        if (len(jobs_queue)) > 0:
            print("&&&&&&&&&fair function overhead: ", f_e - f_s)
        # if violator_id != len(jobs_queue):
        #     # print(violator_id)
        #     # print(len(jobs_queue))
        #     # print("start aggressive drop!!!!!!!!!!")
        #     aggressive_drop(violator_id, jobs_queue, wq_overhead, database)

        # print("overhead: ", e - s)
        with work_queue_lock:
            wq_overhead = work_queue_overhead.value
        start_time = time.time_ns() / 1000000 + 3
        with last_job_lock:
            if last_job_start_time.value > 0:
                start_time = last_job_start_time.value
        wait = wq_overhead - (time.time_ns()/1000000 - start_time)
        interval = max(0.002, wait * 0.7 / 1000 - solver_lat/1000)
        # interval = max(0.001, wait * 0.75 / 1000)
        # interval = 0.015
        time.sleep(interval)

