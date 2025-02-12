import time
from policy import *

def queue_watcher(recv_queue, send_queue, database, estimate_drift,
        drift_lock, interval=0.03):

    # while True:
    #     while not recv_queue.empty():
    #         print("nani\n\n")
    #         j = recv_queue.get()
    #         send_queue.put(j)

    # 1. every interval, look at the queue to see if there is ddl violation 
    start_clock = time.time_ns() / 1000000
    jobs_queue = []

    while True:
        num_jobs = 0
        jobs_buffer = []
        while not recv_queue.empty() and num_jobs < 30:
            j = recv_queue.get()
            jobs_buffer.append(j)
            num_jobs += 1

        # TODO: calc overhead, put jobs to consumer queue
        if len(jobs_queue) > 0:
            j = jobs_queue.pop(0)
            send_queue.put(j)

        jobs_queue = jobs_queue + jobs_buffer
        jobs_queue.sort(key=lambda x: x.ddl)

        window = detect_delay(jobs_queue)

        # TODO calc optimization overhead, put jobs into send_queue

        drift = 0
        with drift_lock:
            drift = estimate_drift.value

        # TODO optimization
        aggressive_drop(window, database)

        with drift_lock:
            estimate_drift.value -= drift
        # print("overhead: ", e - s)
        time.sleep(0.0002)


