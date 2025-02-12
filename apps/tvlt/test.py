# TODO: modality straggler exist?
# TODO: compiler into one model?

import os
import json
import copy
import statistics
import time
import random

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

from torch.profiler import profile, record_function, ProfilerActivity

# =============== user-defined ===============


class MoseiInfer(Inference):
    
    def forward(self, model, batch):
        encoder_last_hidden_outputs, *_ = model(video=batch["video_data"], audio=batch["audio_data"])
        sentiment_score = model.classifier(encoder_last_hidden_outputs).squeeze().data
        return sentiment_score

class MoseiInputManager(InputManager):

    def split_batch(self, batch, split_sections=None):
        """
        Define how to split a given batch. By default, the function takes
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
 
def get_mosei_config():
    mosei_config = {'audio_only': False,
        'audio_patch_size': [16, 16],
        'audio_size': 1024,
        'batch_size': 32,
        # 'batch_size': 8,
        'data_root': './dataset/cmumosei/',
        'datasets': ['mosei'],
        'decay_power': 1,
        'decoder_hidden_size': 512,
        'draw_false_audio': 0,
        'draw_false_text': 0,
        'draw_false_video': 0,
        'drop_rate': 0.1,
        'exp_name': 'cls_mosei',
        'fast_dev_run': False,
        'frame_masking': False,
        'frequency_size': 128,
        'get_va_recall_metric': False,
        'hidden_size': 768,
        'learning_rate': 1e-05,
        'load_hub_path': '',
        'load_local_path': '',
        'log_dir': 'result',
        'loss_names': {'mae_audio': 0,
                    'mae_video': 0,
                    'mlm': 0,
                    'mosei': 1,
                    'moseiemo': 0,
                    'vam': 0,
                    'vqa': 0,
                    'vtm': 0},
        'max_audio_patches': 1020,
        'max_epoch': 10,
        'max_frames': 64,
        'max_steps': 10,
        'max_text_len': 40,
        'mlm_prob': 0.15,
        'mlp_ratio': 4,
        'model_type': 'mae_vit_base_patch16_dec512d8b',
        'num_frames': 8,
        'num_gpus': 2,
        'num_heads': 12,
        'num_layers': 12,
        'num_nodes': 1,
        # 'num_workers': 16,
        'num_workers': 0,
        'optim_type': 'adamw',
        'patch_size': 16,
        'per_gpu_batchsize': 32,
        # 'per_gpu_batchsize': 8,
        'seed': 0,
        'strict_load': False,
        'test_only': False,
        'tokenizer': 'bert-base-uncased',
        'use_audio': True,
        'use_mae': False,
        'use_text': False,
        'use_video': True,
        'val_check_interval': 0.2,
        'video_only': False,
        'video_size': 224,
        'vocab_size': 30522,
        'vqav2_label_size': 3129,
        'warmup_steps': 100,
        'weight_decay': 0.001,
        'whole_word_masking': False}

    return mosei_config

def warm_up(model, dataloader, mosei_infer, warmup_steps=3, device=0):
    with torch.no_grad():
        model.eval()
        for i_batch, sample_batched in enumerate(dataloader):
            sample_batched = dict_to(sample_batched, device=device)
            _ = mosei_infer(model, sample_batched)
            if i_batch + 1 >= warmup_steps:
                break




def mp_inference_round_robin_helper(
        model,
        dataloader,
        mosei_input_manager,
        mosei_infer,
        splitted,
        access_order,
        device,
        proc_id):
        
    with torch.no_grad():
        model.eval()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        print(access_order)
        

        
        # create streams
        streams = []
        num_streams = 3
        for i in range(num_streams):
            streams.append(torch.cuda.Stream(device=device))

        stream = streams[proc_id]

        num_trials = 5
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.cuda.stream(stream):
                starter.record()
                for i in range(num_trials):
                    for j in access_order:
                        sub_batch = splitted[j]
                        model_input = copy_dict_to(sub_batch, device=device)
                        _ = mosei_infer(model, model_input)
                ender.record()
                ender.synchronize()
        prof.export_chrome_trace("trace_" + str(proc_id) + ".json")
        print("proc stream time: {}", starter.elapsed_time(ender))




def mp_inference_fixed_mod_helper(
        model,
        mosei_input_manager,
        mosei_infer,
        mod_data,
        device,
        proc_id
        ):
    with torch.no_grad():
        model.eval()

        streams = []
        for i in range(3):
            streams.append(torch.cuda.Stream(device=device))

        stream = streams[proc_id]
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.cuda.stream(stream):
                print(torch.cuda.current_stream())
                starter.record()
                
                num_trials = 15
                for i in range(num_trials):
                    model_input = copy_dict_to(mod_data, device=device)
                    _ = mosei_infer(model, model_input)

                ender.record()
                ender.synchronize()

        prof.export_chrome_trace("trace_" + str(proc_id) + ".json")
        print("proc stream time: {}", starter.elapsed_time(ender))
        print(time.time_ns())



def test_continuous_copy_inf_order(
        model,
        dataloader,
        mosei_input_manager,
        mosei_infer,
        drop_policy,
        device,
        num_streams=3,
        test_id=0):
        
    with torch.no_grad():
        model.eval()
        print("Begin warmup..")
        warm_up(model, dataloader, mosei_infer, device=device)

        sample_batched = next(iter(dataloader))

        splitted = mosei_input_manager.split_batch(sample_batched,
                    split_sections=list(drop_policy.values())) 

        for i, (drop_mod_name, drop_size) in enumerate(drop_policy.items()):
            splitted[i] = mosei_input_manager.remove_mod(splitted[i],
                    removed_modality=drop_mod_name.split(","))

        # torch.cuda.reset_max_memory_allocated(device=device)
        # starter.record(stream=torch.cuda.current_stream(device=device))


        # create streams
        streams = []
        num_trials = 30
        if test_id == 0:
            num_streams = len(splitted)
            cuda_starters = [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
            cuda_enders= [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
            for i in range(num_streams):
                streams.append(torch.cuda.Stream(device=device))


            start = time.time_ns()
            # start stream timer
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    cuda_starters[i].record()

            for _ in range(num_trials):
                # TODO: shuffle splitted
                for i, sub_batch in enumerate(splitted):
                    # TODO data movement
                    if test_id == 0: # every micro-batch uses a dedicated stream
                        with torch.cuda.stream(streams[i]):
                            model_input = copy_dict_to(sub_batch, device=device)
                            _ = mosei_infer(model, model_input)

            # end stream timer
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    cuda_enders[i].record()


            synchronize_gpus(devices=[device])
            end = time.time_ns()
            print("total time: ", (end - start) / 1000)

            for i in range(num_streams):
                print("stream {}: {}", i, cuda_starters[i].elapsed_time(cuda_enders[i]))
        elif test_id == 1:
            num_streams = len(splitted)
            cuda_starters = [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
            cuda_enders= [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
            for i in range(num_streams):
                streams.append(torch.cuda.Stream(device=device))


            start = time.time_ns()
            # start stream timer
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    cuda_starters[i].record()

            for _ in range(num_trials):
                # TODO: shuffle splitted
                splitted.reverse()
                for i, sub_batch in enumerate(splitted):
                    # TODO data movement
                    if test_id == 1: # every micro-batch uses a dedicated stream
                        with torch.cuda.stream(streams[i]):
                            model_input = copy_dict_to(sub_batch, device=device)
                            _ = mosei_infer(model, model_input)

            # end stream timer
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    cuda_enders[i].record()


            synchronize_gpus(devices=[device])
            end = time.time_ns()
            print("total time: ", (end - start) / 1000)

            for i in range(num_streams):
                print("stream {}: {}", i, cuda_starters[i].elapsed_time(cuda_enders[i]))
        elif test_id == 2:
            num_streams = len(splitted)
            cuda_starters = [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
            cuda_enders= [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
            for i in range(num_streams):
                streams.append(torch.cuda.Stream(device=device))


            start = time.time_ns()
            # start stream timer
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    cuda_starters[i].record()

            for _ in range(num_trials):
                # TODO: shuffle splitted
                splitted = random.shuffle(splitted)
                for i, sub_batch in enumerate(splitted):
                    # TODO data movement
                    if test_id == 2: # every micro-batch uses a dedicated stream
                        with torch.cuda.stream(streams[i]):
                            model_input = copy_dict_to(sub_batch, device=device)
                            _ = mosei_infer(model, model_input)

            # end stream timer
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    cuda_enders[i].record()


            synchronize_gpus(devices=[device])
            end = time.time_ns()
            print("total time: ", (end - start) / 1000)

            for i in range(num_streams):
                print("stream {}: {}", i, cuda_starters[i].elapsed_time(cuda_enders[i]))







if __name__ == '__main__':
    # This is necessary to bypass librosa limitation
    import torch.multiprocessing as mp
    mp.set_start_method("forkserver")
    # mp.set_start_method("spawn")
    import joblib
    import sklearn


    
    def my_main():
        from model.modules import Transformer
        from model.data.datamodules.multitask_datamodule import MTDataModule

        print("yeah!")

        # context setup
        _config = get_mosei_config()
        pl.seed_everything(_config["seed"])

        # dataset setup
        dm = MTDataModule(_config, dist=False)
        dm.setup("test")
        dm.prepare_data()


        device = 0
        torch.cuda.set_per_process_memory_fraction(1.0)
        # model setup
        model = MOSEI_sentiment_model()
        model.eval()
        # model = torch.compile(model, fullgraph=True)
        model = model.to(device=device)
        exp_name = f'{_config["exp_name"]}'

        # infer setup
        mosei_infer = MoseiInfer()
        mosei_input_manager = MoseiInputManager(
            modality_names = {
                "video": ["video_data"],
                "audio": ["audio_data"]
            }
        )

    
        # candidates = load_candidates(filename="./candidates_result/mosei_32.json")
        # drop_policy = candidates["0.73"][0]
        # drop_policy = {"None": 27, "video": 5}
        # drop_policy = {"video": 18, "None": 14}
        drop_policy = {"video": 5, "audio": 7,  "None": 20}
        # drop_policy = {"video": 3, "audio": 4, "None": 1}
        # exec_policies = [
        #         {0: ["video"], 1: ["audio"], 2: ["None"]},
        #         {0: ["video"], 1: [], 2: ["audio", "None"]},
        #         {0: [], 1: ["video", "audio"], 2: ["None"]},
        #         {0: [], 1: ["video"], 2: ["audio", "None"]},
        #         {0: [], 1: [], 2: ["video", "audio", "None"]},
        #         ]

        # exec_policies = [
        #         [0, 2],
        #         [1, 1]
        #         ]

        exec_policies = [[1, 1, 1]]
        # exec_policies = [
        #         [1, 1, 1],
        #         [1, 0, 2],
        #         [0, 2, 1],
        #         [0, 1, 2],
        #         [0, 0, 3]
        #         ]

        print(drop_policy)
        # infer(model, dm.test_dataloader(), mosei_input_manager, mosei_infer, drop_policy, exec_policies, device=device)
        # default_infer(model, dm.test_dataloader(), mosei_infer, device=device)
        # explore_policies(model, dm.test_dataloader(), mosei_input_manager, mosei_infer, drop_policy, exec_policies,
        #         sample_iter=5, device=device)
        # test_continuous_copy(model, dm.test_dataloader(), mosei_input_manager,
        #         mosei_infer, drop_policy, device)
        # mp_inference_no_check(model, dm.test_dataloader(), mosei_input_manager,
        #         mosei_infer, drop_policy, device)
        # test_continuous_copy_inf_order(model, dm.test_dataloader(), mosei_input_manager,
        #          mosei_infer, drop_policy, device, num_streams=3, test_id=1)
        mp_inference_round_robin(model, dm.test_dataloader(), mosei_input_manager,
                mosei_infer, drop_policy, device)
        # mp_inference_fixed_mod(model, dm.test_dataloader(), mosei_input_manager,
        #         mosei_infer, drop_policy, device)
        
        
        
    
    


    def mp_inference_round_robin(
            model,
            dataloader,
            mosei_input_manager,
            mosei_infer,
            drop_policy,
            device=0):

        with torch.no_grad():
            model.eval()
            print("before warm up")
            warm_up(model, dataloader, mosei_infer, device=device)
            num_proc = 3
            print("after warm up but before share memory")
            model.share_memory()
            sample_batched = next(iter(dataloader))

            splitted = mosei_input_manager.split_batch(sample_batched,
                            split_sections=list(drop_policy.values())) 

            num_proc = len(splitted)

            for i, (drop_mod_name, drop_size) in enumerate(drop_policy.items()):
                splitted[i] = mosei_input_manager.remove_mod(splitted[i],
                        removed_modality=drop_mod_name.split(","))


            # copy splitted
            splitted_all = [copy.deepcopy(splitted) for _ in range(len(splitted))]

            processes = []
            print("finished warm up!!")
            start = time.time_ns()
            for rank in range(num_proc):
                if rank == 0:
                    access_order = [0, 1, 2]
                elif rank == 1:
                    access_order = [1, 2, 0]
                elif rank == 2:
                    access_order = [2, 1, 0]

                p = mp.Process(target=mp_inference_round_robin_helper, args=(
                    model, dataloader, mosei_input_manager,
                    mosei_infer, splitted_all[rank], access_order, device, rank
                    ))
                print("process starts..")
                p.start()
                print("process ends..")
                processes.append(p)
            for p in processes:
                p.join()

            synchronize_gpus(devices=[device])
            end = time.time_ns()
            print("time: {}", (end-start)/1000/1000)




    def mp_inference_fixed_mod(
            model,
            dataloader,
            mosei_input_manager,
            mosei_infer,
            drop_policy,
            device=0):
        """
        Each process only handles one single type of micro-batch
        e.g. proc 1 only do video-only batches
        """

        with torch.no_grad():
            model.eval()
            print("before warm up")
            warm_up(model, dataloader, mosei_infer, device=device)
            print("after warm up but before share memory")
            model.share_memory()

            sample_batched = next(iter(dataloader))

            splitted = mosei_input_manager.split_batch(sample_batched,
                        split_sections=list(drop_policy.values())) 

            num_proc = len(splitted)

            for i, (drop_mod_name, drop_size) in enumerate(drop_policy.items()):
                splitted[i] = mosei_input_manager.remove_mod(splitted[i],
                        removed_modality=drop_mod_name.split(","))



            processes = []
            print("finished warm up!!")
            start = time.time_ns()

            for rank in range(num_proc):

                p = mp.Process(target=mp_inference_fixed_mod_helper, args=(
                    model, mosei_input_manager,
                    mosei_infer, splitted[rank], device, rank
                    ))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            
            end = time.time_ns()
            print("time: {}", (end-start)/1000/1000)



        


    def explore_policies(model, dataloader, mosei_input_manager, mosei_infer, drop_policy, exec_policies,
            sample_iter=5, device=0):

        # warm_up(model, dataloader, mosei_infer, device=device)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        all_stats = []

        # inference
        with torch.no_grad():
            model.eval()
            for exec_policy in exec_policies:
                exec_policy_total_lat = 0
                exec_policy_total_max_mem = 0
                
                starter.record(stream=torch.cuda.current_stream(device=device))




                for i_batch, sample_batched in enumerate(dataloader):
                    splitted = mosei_input_manager.split_batch(sample_batched,
                            split_sections=list(drop_policy.values())) 

                    for i, (drop_mod_name, drop_size) in enumerate(drop_policy.items()):
                        splitted[i] = mosei_input_manager.remove_mod(splitted[i],
                                removed_modality=drop_mod_name.split(","))
                        # splitted[i] = dict_to(splitted[i], device=device)

                    batches = []
                    
                    # torch.cuda.reset_max_memory_allocated(device=device)
                    # starter.record(stream=torch.cuda.current_stream(device=device))

                    while temp < 10:
                        # TODO: shuffle splitted
                        for i, sub_batch in enumerate(splitted):
                            # data movement
                            model_input = copy_dict_to(sub_batch, device=device)
                            batches.append(model_input)
                            num_to_exec = exec_policy[i]
                            for j in range(num_to_exec):
                                b = batches.pop(0)
                                _ = mosei_infer(model, b)
             
                    
                    temp = 0
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        while temp < 10:
                            # TODO: shuffle splitted
                            for i, sub_batch in enumerate(splitted):
                                # data movement
                                model_input = dict_to(sub_batch, device=device)
                                batches.append(model_input)
                                num_to_exec = exec_policy[i]
                                for j in range(num_to_exec):
                                    b = batches.pop(0)
                                    _ = mosei_infer(model, b)
                    prof.export_chrome_trace("trace.json")

                    # ender.record(stream=torch.cuda.current_stream(device=device))
                    # torch.cuda.synchronize(device=device)
                    # exec_policy_total_lat += starter.elapsed_time(ender)
                    # exec_policy_total_max_mem += torch.cuda.max_memory_allocated(device=device)

                    if i_batch + 1 >= sample_iter:
                        break

                ender.record(stream=torch.cuda.current_stream(device=device))
                torch.cuda.synchronize(device=device)
                # print(starter.elapsed_time(ender))
                
                print("policy finished")
                print()


                # avg_lat = exec_policy_total_lat / sample_iter 
                # avg_max_mem = exec_policy_total_max_mem / sample_iter
                # all_stats.append((avg_lat, avg_max_mem))
                all_stats.append(starter.elapsed_time(ender))

            print(all_stats)



                     



    def infer(model, dataloader, mosei_input_manager, mosei_infer, drop_policy, device=0):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # inference
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                splitted = mosei_input_manager.split_batch(sample_batched,
                        split_sections=list(drop_policy.values())) 

                for i, (drop_mod_name, drop_size) in enumerate(drop_policy.items()):
                    splitted[i] = mosei_input_manager.remove_mod(splitted[i],
                            removed_modality=drop_mod_name.split(","))
                    splitted[i] = dict_to(splitted[i], device=device)

                
                torch.cuda.reset_max_memory_allocated(device=device)
                starter.record(stream=torch.cuda.current_stream(device=device))
                for sub_batch in splitted:
                    _ = mosei_infer(model, sub_batch)
                ender.record(stream=torch.cuda.current_stream(device=device))
                torch.cuda.synchronize(device=device)
                starter.elapsed_time(ender)
                max_mem = torch.cuda.max_memory_allocated(device=device)


    def default_infer(model, dataloader, mosei_infer, device=0):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                sample_batched = dict_to(sample_batched, device=device)
                torch.cuda.reset_max_memory_allocated(device=device)
                starter.record(stream=torch.cuda.current_stream(device=device))
                _ = mosei_infer(model, sample_batched)
                ender.record(stream=torch.cuda.current_stream(device=device))
                torch.cuda.synchronize(device=device)
                print(starter.elapsed_time(ender))
                print(torch.cuda.max_memory_allocated(device=device))




    my_main()
