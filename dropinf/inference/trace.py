from multiprocessing import Process, Value, Array, Manager
import os
import time
import numpy as np

class Tracer:

    def __init__(
            self,
            lock,
            submitted_latency,
            submitted_accuracy,
            processed_latency,
            processed_accuracy,
            file_name):
        self.lock = lock
        self.submitted_jobs = Value("i", 0)
        self.submitted_requests = Value("i", 0)
        self.submitted_latency = submitted_latency
        self.submitted_accuracy = submitted_accuracy

        self.processed_jobs = Value("i", 0)
        self.processed_requests = Value("i", 0)
        self.processed_latency = processed_latency
        self.processed_accuracy = processed_accuracy

        self.num_modified = Value("i", 0)
        
        self.current_sub_job_id = Value("i", 0)
        self.current_proc_job_id = Value("i", 0)

        self.file_name = file_name
        self.init_trace_file()

        # test specific attributes
        self.avg_violation_ratio = Value("d", 0)

    def init_trace_file(self):
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        # open(self.file_name, "w+").close()
        open(self.file_name, "w").close()
        with open(self.file_name, "a") as f:
            f.write("# id submitted_jobs submitted_requests " + \
                    "submitted_avg_latency submitted_latency_std " + \
                    "submitted_avg_accuracy submitted_accuracy_std " + \
                    "processed_jobs processed_requests " + \
                    "processed_avg_latency processed_latency_std " + \
                    "processed_avg_accuracy processed_accuracy_std " + \
                    "num_modified\n")

    def reset(self):
        # with self.lock:
        self.submitted_jobs.value = 0
        self.submitted_requests.value = 0
        self.submitted_latency[:] = []
        self.submitted_accuracy[:] = []

        self.processed_jobs.value = 0
        self.processed_requests.value = 0
        self.processed_latency[:] = []
        self.processed_accuracy[:] = []

        self.num_modified.value = 0

def write_trace(tracer,
                index,
                submitted_jobs,
                submitted_requests,
                submitted_latency,
                submitted_accuracy,
                processed_jobs,
                processed_requests,
                processed_latency,
                processed_accuracy,
                num_modified):
    
    submitted_avg_latency = np.mean(submitted_latency)
    submitted_latency_std = np.std(submitted_latency)
    submitted_avg_accuracy = np.mean(submitted_accuracy)
    submitted_accuracy_std = np.std(submitted_accuracy)

    processed_avg_latency = np.mean(processed_latency)
    processed_latency_std = np.std(processed_latency)
    processed_avg_accuracy = np.mean(processed_accuracy)
    processed_accuracy_std = np.std(processed_accuracy)

    line = str(index) + " " + \
            str(submitted_jobs) + " " + \
            str(submitted_requests) + " " + \
            str(submitted_avg_latency) + " " + \
            str(submitted_latency_std) + " " + \
            str(submitted_avg_accuracy) + " " + \
            str(submitted_accuracy_std) + " " + \
            str(processed_jobs) + " " + \
            str(processed_requests) + " " + \
            str(processed_avg_latency) + " " + \
            str(processed_latency_std) + " " + \
            str(processed_avg_accuracy) + " " + \
            str(processed_accuracy_std) + " " + \
            str(num_modified) + "\n"
    
    with open(tracer.file_name, "a") as f:
        f.write(line)

def trace_proc(tracer):
    time.sleep(5)
    with tracer.lock:
        tracer.reset()

    index = 0
    zero_counter = 0
    while True:
        with tracer.lock:
            submitted_jobs = tracer.submitted_jobs.value
            submitted_requests = tracer.submitted_requests.value
            submitted_latency = list(tracer.submitted_latency)
            submitted_accuracy = list(tracer.submitted_accuracy)

            processed_jobs = tracer.processed_jobs.value
            processed_requests = tracer.processed_requests.value
            processed_latency = list(tracer.processed_latency)
            processed_accuracy = list(tracer.processed_accuracy)

            num_modified = tracer.num_modified.value

            tracer.reset()
        
        write_trace(
            tracer,
            index,
            submitted_jobs,
            submitted_requests,
            submitted_latency,
            submitted_accuracy,
            processed_jobs,
            processed_requests,
            processed_latency,
            processed_accuracy,
            num_modified
        )

        time.sleep(4)

        if submitted_jobs <= 0 and submitted_requests <= 0:
            zero_counter += 1

        if zero_counter >= 3:
            print("tracer exiting")
            return

        index += 1

def collect_violation_ratio(tracer):
    time.sleep(5)
    with tracer.lock:
        tracer.reset()

    index = 0
    zero_counter = 0
    total_violation_ratio = []
    while True:
        with tracer.lock:
            submitted_jobs = tracer.submitted_jobs.value
            submitted_requests = tracer.submitted_requests.value
            submitted_latency = list(tracer.submitted_latency)
            submitted_accuracy = list(tracer.submitted_accuracy)

            processed_jobs = tracer.processed_jobs.value
            processed_requests = tracer.processed_requests.value
            processed_latency = list(tracer.processed_latency)
            processed_accuracy = list(tracer.processed_accuracy)

            num_modified = tracer.num_modified.value

            tracer.reset()

        if processed_requests > 0 and submitted_requests > 0:
            print("nani????????????????????\n\n")
            on_time_ratio = processed_requests / submitted_requests
            if on_time_ratio > 1:
                on_time_ratio = 1
            elif on_time_ratio < 0:
                on_time_ratio = 0
            
            violation_ratio = 1 - on_time_ratio
            total_violation_ratio.append(violation_ratio)
        elif processed_requests <= 0 and submitted_requests > 0:
            violation_ratio = 0.99
            total_violation_ratio.append(violation_ratio)
        elif processed_requests > 0 and submitted_requests <= 0:
            total_violation_ratio.append(0.01)


        time.sleep(4)

        if submitted_jobs <= 0 and submitted_requests <= 0:
            zero_counter += 1

        if zero_counter >= 3:
            print("tracer exiting")
            print(total_violation_ratio)
            with tracer.lock:
                tracer.avg_violation_ratio.value = sum(total_violation_ratio) / len(total_violation_ratio)
            return

        index += 1