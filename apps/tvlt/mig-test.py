import os
import json
import copy
import statistics
import time
import random
import argparse

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


def infer(
        model,
        dataloader,
        mosei_input_manager,
        mosei_infer,
        drop_policy, # drop_policy ordered based on process
        device,
        trials=15
        ):
    
    with torch.no_grad():
        model.eval()

        # warm_up(model, dataloader, mosei_infer, warmup_steps=2, device=device)
        
        # warmup not needed?
        sample_batched = next(iter(dataloader))
        splitted = mosei_input_manager.split_batch(sample_batched,
                    split_sections=list(drop_policy.values())) 
        for i, (drop_mod_name, drop_size) in enumerate(drop_policy.items()):
            splitted[i] = mosei_input_manager.remove_mod(splitted[i],
                    removed_modality=drop_mod_name.split(","))



        splitted.pop()
        splitted.pop()
        trials *= 3



        # submit based on given modality order
        start = time.time_ns()
        for i in range(trials):
            for micro_batch in splitted:
                micro_batch = copy_dict_to(micro_batch, device=device)
                _ = mosei_infer(model, micro_batch)

        # sync?
        synchronize_gpus(devices=[device])
        end = time.time_ns()
        print("total time: ", (end - start) / 1000000)




if __name__ == '__main__':
    # This is necessary to bypass librosa limitation
    # import torch.multiprocessing as mp
    # mp.set_start_method("forkserver")

    total_start = time.time_ns()

    import joblib
    import sklearn
    from model.modules import Transformer
    from model.data.datamodules.multitask_datamodule import MTDataModule

    device = torch.cuda.current_device()


    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--id", type=int, help="proc id")
    args = argParser.parse_args()

    # context setup
    _config = get_mosei_config()
    pl.seed_everything(_config["seed"])

    # dataset setup
    dm = MTDataModule(_config, dist=False)
    dm.setup("test")
    dm.prepare_data()
    model = MOSEI_sentiment_model(device=device)
    model.eval()
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

    drop_policies = [
            {"video": 5, "audio": 7, "None": 20},
            {"audio": 7, "None": 20, "video": 5},
            {"None": 20, "video": 5, "audio": 7},
            ]

    print(args.id)
    print(drop_policies[args.id])

    infer(
            model,
            dm.test_dataloader(),
            mosei_input_manager,
            mosei_infer,
            drop_policies[args.id],
            device
            )

    total_end = time.time_ns()
    print("total lat: ", (total_end - total_start) / 1000000)






