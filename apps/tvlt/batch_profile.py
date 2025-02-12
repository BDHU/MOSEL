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
from dropinf.utils.file import *
from dropinf.utils.solver import *
from config import *

from torch.profiler import profile, record_function, ProfilerActivity

# =============== user-defined ===============


class MoseiInfer(Inference):

    def forward(self, model, batch, args):
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

if __name__ == '__main__':
    # This is necessary to bypass librosa limitation
    import torch.multiprocessing as mp
    mp.set_start_method("forkserver")
    # mp.set_start_method("spawn")
    import joblib
    import sklearn

    def batch_profile():
        from model.modules import Transformer
        from model.data.datamodules.multitask_datamodule import MTDataModule

        # context setup
        _config = get_mosei_config()
        pl.seed_everything(_config["seed"])

        # dataset setup
        dm = MTDataModule(_config, dist=False)
        dm.setup("test")
        dm.prepare_data()

        # torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cudnn.allow_tf32 = False


        # device = 0
        device = 0
        quantize = False
        save_dir = "./mosei_batch_profiling_result/mosei_batch_tf32.json"
        # model setup
        model = MOSEI_sentiment_model(device=device)
        model.eval()
        if quantize is True:
            model = model.half()
            # model = torch.quantization.quantize_dynamic(
            #     model,  # the original model
            #     {torch.nn.Linear},  # a set of layers to dynamically quantize
            #     dtype=torch.qint8)
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



        profiler = Profiler(
            num_nodes=_config["num_nodes"],
            accelerator="gpu",
            batch_size=_config["batch_size"],
            infer_func=mosei_infer,
            input_manager=mosei_input_manager
        )

        profiler.profile_batch(min_batch=1, max_batch=32, model=model,
                dataloader=dm.test_dataloader(), device=device,
                preprocess=False,
                quantize=quantize,
                save_dir=save_dir)

    batch_profile()