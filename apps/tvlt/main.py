import os
import json
import copy
import statistics

import pytorch_lightning as pl
from torchmetrics import Metric

# from model.config import ex
from model.modules import Transformer
from model.data.datamodules.multitask_datamodule import MTDataModule
import torch
import numpy as np

from demos import MOSEI_sentiment_model

from dropinf.inference.inference import Inference
from dropinf.profiler.profiler import Profiler
from dropinf.inference.input import *


def get_mosei_config():
    mosei_config = {'audio_only': False,
        'audio_patch_size': [16, 16],
        'audio_size': 1024,
        'batch_size': 32,
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
        'num_workers': 16,
        'optim_type': 'adamw',
        'patch_size': 16,
        'per_gpu_batchsize': 32,
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




def a2_parse(a):
    if a < 0:
        res = 0
    else:
        res = 1
    return res

def get_logits_a2(score):
    eyes = torch.eye(2).to(score.device)
    score = torch.tensor([a2_parse(item) for item in score]).to(score.device)
    return eyes[score]


class MoseiAccuracy(Metric):
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("result", default=torch.tensor([]), dist_reduce_fx="cat")

    # def update(self, logits, target): 
    def update(self, output, batch): 
        logits, target = get_logits_a2(output), torch.tensor(batch["label2"])
    
        logits, target = (
            torch.tensor(logits.detach().cpu().numpy()).to(self.correct.device),
            torch.tensor(target.detach().cpu().numpy()).to(self.correct.device),
        )
        if logits.size(1) > 1:
            preds = logits.argmax(dim=-1).squeeze()
        else:
            preds = (logits >= 0).squeeze()
        target = target.squeeze()    
        preds = preds.to(target.device)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.result = torch.cat((self.result, preds == target))

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

    def stdev(self):
        return torch.std(self.result)


class MoseiInfer(Inference):
    
    def forward(self, model, batch):
        encoder_last_hidden_outputs, *_ = model(video=batch["video_data"], audio=batch["audio_data"])
        sentiment_score = model.classifier(encoder_last_hidden_outputs).squeeze().data
        return sentiment_score


# @ex.automain
def main():
    # context setup
    # _config = copy.deepcopy(_config)
    _config = get_mosei_config()
    pl.seed_everything(_config["seed"])

    # dataset setup
    dm = MTDataModule(_config, dist=False)
    dm.setup("test")
    dm.prepare_data()

    # model setup
    model = MOSEI_sentiment_model()
    model.eval()
    exp_name = f'{_config["exp_name"]}'

    # infer setup
    mosei_infer = MoseiInfer()
    mosei_input_manager = InputManager(
        modality_names = {
            "video": ["video_data"],
            "audio": ["audio_data"]
        }
    )

    # test run
    # for a in dm.test_dataloader():
    #     print(a)
    #     mosei_infer(model=model, batch=a)
    #     break

    # profiler setup
    profiler = Profiler(
        num_nodes=_config["num_nodes"],
        accelerator="gpu",
        batch_size=_config["batch_size"],
        infer_func=mosei_infer,
        input_manager=mosei_input_manager
    )

    metric = MoseiAccuracy()
    profiler.register_metric(name="accuracy", metric=metric)

    print("start profiling...")
    profiler.profile(model=model, dataloader=dm.test_dataloader(), device=0)
    print("profiling ended")

if __name__ == "__main__":
    main()
