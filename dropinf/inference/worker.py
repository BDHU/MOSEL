import torch
import time
import os
from dropinf.utils.cuda import *

def worker(model, infer, queue, recv_queue, input_manager, remove_mod, work_queue_overhead,
        work_queue_lock, last_job_start_time, last_job_lock, tracer, quantize, device, args):
    # prepare model
    # model = MOSEI_sentiment_model(device=device)
    # model = torch.compile(model, fullgraph=True)
    with torch.no_grad():
        model.eval()
        model = model.to(device=device)
    # mosei_infer = MoseiInfer()
    mosei_infer = infer
    # time.sleep(1)

    with torch.no_grad():
        model.eval()
        
        j_n = 0
        while True:
            if queue.empty():
                with work_queue_lock:
                    work_queue_overhead.value = 0
                with last_job_lock:
                    last_job_start_time.value = 0
            job = queue.get()

            if job.debug == "end":
                recv_queue.put("end")
                continue
            j_n += 1
            start = 0.0
            with last_job_lock:
                last_job_start_time.value = time.time_ns() / 1000000
                start = last_job_start_time.value
            
            
            print("worker job status:----------------")
            print("job id: ", job.job_id)
            print("start time: ", start)
            print(job.drop_policy)
            print("job latency: ", job.comp_time)
            print("job ddl: ", job.ddl)
            print("job estimated finish time: ", start + job.comp_time)



            # if start + job.comp_time > job.ddl + 50: # fair
            if start + job.comp_time > job.ddl + 12: # random
            # if start + job.comp_time > job.ddl: # aggressive
                with work_queue_lock:
                    work_queue_overhead.value -= job.comp_time
                print("dropped by worker\n\n\n")
                continue
            if remove_mod is True:
                splitted = input_manager.split_batch(job.payload,
                        split_sections=list(job.drop_policy.values()))
                for i, (drop_mod_name, drop_size) in enumerate(
                        job.drop_policy.items()):
                    splitted[i] = input_manager.remove_mod(splitted[i],
                            removed_modality=drop_mod_name.split(","))
                job.payload = splitted
            else:
                pass
            

            # # metric collect
            # prev_sec_range = -1
            # if job.sec_range > prev_sec_range:
            #     jobs_results.write(str(prev_sec_range) + " " + str(total_jobs_per_sec) + "\n")
            #     requests_results.write(str(prev_sec_range) + " " + str(total_requests_per_sec) +  "\n")
            #     total_jobs_per_sec = 0
            #     total_requests_per_sec = 0
            #     prev_sec_range = job.sec_range
            # elif job.sec_range == prev_sec_range:
            #     total_jobs_per_sec += 1
            #     total_requests_per_sec += job.batch_size





            all_results = []
            for b in job.payload:
                b = copy_dict_to(b, device=device)
                r = mosei_infer(model, b, args)
                all_results.append(r.cpu().numpy())
            # recv_queue.put(all_results)
            recv_queue.put(job.job_id)


                        # 1. throughput
            # if job.job_id > 34:
            with tracer.lock:
                tracer.processed_jobs.value += 1
                tracer.processed_requests.value += job.batch_size
                tracer.processed_latency.append(time.time_ns() / 1000000 - start)
                tracer.processed_accuracy.append(job.acc_in_use)
                tracer.current_proc_job_id = job.job_id
            # 2. on-time ratio


            finish_time = time.time_ns() / 1000000
            print("job actual execution time: ", finish_time - start)
            print("\n\n")
            estimated_finished_time = start + job.comp_time
            time_saved = estimated_finished_time - finish_time
            with work_queue_lock:
                work_queue_overhead.value -= (job.comp_time + time_saved)


