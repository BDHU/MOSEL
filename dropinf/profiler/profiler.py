import torch
import copy
import time

from dropinf.profiler.metrics import *
from dropinf.profiler.utils import *
from dropinf.utils.model import model_to
from dropinf.utils.cuda import *
from dropinf.profiler.storage import *
from dropinf.profiler.recorder import *


class Profiler:
    
    def __init__(
        self,
        num_nodes=None,
        accelerator="cpu",
        batch_size=1,
        infer_func=None,
        input_manager=None,
        profiler_result_dump=None
    ):
        self.num_nodes = num_nodes
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.infer = infer_func
        self.input_manager = input_manager 
        self.profiler_result_dump = profiler_result_dump


        # set up
        self.storage = Storage()
        self.custom_metrics = {}

    def register_metric(self, name=None, metric=None):
        self.custom_metrics[name] = metric 

    def __init_recorder_metrics(self, recorder=None, metrics=None):
        assert(recorder is not None)
        set_metrics(recorder)

    
    def __init_recorder(self) -> Recorder:
        recorder = Recorder()
        self.__init_recorder_metrics(recorder=recorder)
        return recorder

    def __warmup(
            self,
            model=None,
            dataloader=None,
            quantize=False,
            warmup_iter=3,
            device=None):
        for i_batch, sample_batched in enumerate(dataloader):
            if quantize is True:
                sample_batched = dict_to_half(sample_batched)
            sample_batched = dict_to(sample_batched, device=device)
            _ = self.infer(model, sample_batched, None)
            if i_batch >= warmup_iter - 1:
                break

    def profile(self, model=None, dataloader=None, device=None):
        # warm up model and accelerator
        with torch.no_grad():
            model.eval()
            model = model_to(model, device=device)
            self.__warmup(model=model, dataloader=dataloader, device=device)
            
            mod_to_drop = self.input_manager.get_mod_to_remove()
            for dm in mod_to_drop:
                profile_result = self.__profile_step(model=model,
                        dataloader=dataloader, dropped=dm, device=device)
                self.storage.add(removed_mod=dm, batch_size=self.batch_size, 
                        record=profile_result)
            
            self.storage.dump(result_dir=self.profiler_result_dump)

    def profile_batch(
            self,
            min_batch=1,
            max_batch=32,
            model=None,
            dataloader=None, 
            device=None,
            preprocess=False,
            quantize=False,
            args=None,
            save_dir="./"):
        """
        Profile metrics for each batch size
        """
        with torch.no_grad():
            model.eval()

            if device != "cpu":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            elif device == "cpu":
                start = None
                end = None
            else:
                raise Exception("unknown device")

            model = model_to(model, device=device)
            self.__warmup(model=model,
                          dataloader=dataloader,
                          quantize=quantize,
                          device=device)

            metrics = {}
            mod_to_drop = self.input_manager.get_mod_to_remove()
            for dm in mod_to_drop:
                if preprocess is False:
                    batch = next(iter(dataloader))
                    dm_metrics = {}
                    
                    for batch_size in range(min_batch, max_batch+1):
                        metric = {}

                        batch_slice = self.input_manager.slice(batch,
                                start_idx=0, end_idx=batch_size)
                        batch_slice = self.input_manager.remove_mod(
                                batch_slice, removed_modality=dm)
                        if quantize is True:
                            batch_slice = dict_to_half(batch_slice)


                        num_trials = 10
                        total_lat = 0
                        for _ in range(num_trials):
                            if device != "cpu":
                                torch.cuda.reset_max_memory_allocated(device=device)
                                start.record()
                            else:
                                start = time.time_ns() / 1000000

                            batch_slice = copy_dict_to(batch_slice, device=device)
                            _ = self.infer(model, batch_slice, args)

                            if device != "cpu":
                                end.record()
                                torch.cuda.synchronize(device=device)
                                latency = start.elapsed_time(end)
                            else:
                                end = time.time_ns() / 1000000
                                latency = end - start
                            total_lat += latency
                            
                            batch_slice = copy_dict_to(batch_slice, device="cpu")

                            
                        metric["latency"] = total_lat / num_trials
                        if device != "cpu":
                            max_mem = torch.cuda.max_memory_allocated(device=device)
                        else:
                            max_mem = 0
                        metric["max_mem"] = max_mem

                        dm_metrics[0] = {"latency": 0, "max_mem": 0}
                        dm_metrics[batch_size] = metric

                    if len(dm) <= 0:
                        m_key = "None"
                    else:
                        m_key = ','.join(dm)
                    metrics[m_key] = dm_metrics
                        
                else:
                    raise NotImplementedError("Including preprocessing \
                            overhead not yet implemented yet")


            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w+") as fp:
                json.dump(metrics, fp, indent=4)


    r"""
    Perform profile for each batch
    """
    def __profile_step(self, model=None, dataloader=None, dropped=None,
            recorder=None, device=None):
        with torch.no_grad():
            model.eval()
            recorder = self.__init_recorder()
            # submit sub-batches
            for i_batch, sample_batched in enumerate(dataloader):
                # TODO latency 
                batch = self.input_manager.remove_mod(sample_batched,
                        removed_modality=dropped)
                batch = dict_to(batch, device=device)
                torch.cuda.reset_max_memory_allocated()
                recorder.timer_start()
                
                output = self.infer(model, batch)

                recorder.timer_end()
                latency = recorder.timer_elapsed()
                getattr(recorder, f"latency")(
                    lat=latency
                )
                getattr(recorder, f"memory")(
                    mem=torch.cuda.max_memory_allocated()
                )
                getattr(recorder, f"accuracy")(
                    output=output, batch=sample_batched
                )
                # if i_batch >= 3:
                #     break
            return recorder
