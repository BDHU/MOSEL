from typing import Optional
from dropinf.profiler.profiler import Profiler
from dropinf.profiler.recorder import Recorder

class Inference:
    
    def __init__(self, profiler=None):
        if profiler is not None:
            assert(isinstance(profiler, Profiler))
        self._profiler = profiler

    def __call__(self, model, batch, args):
        return self.forward(model, batch, args)

    @property
    def profiler(self):
        return self._profiler

    @profiler.setter
    def set_profiler(self, prof):
        self._profiler = prof

    def register_recorder(self, recorder=None):
        if recorder is None or not isinstance(recorder, Recorder):
            raise ValueError("Can't register None type recorder to infer object")

        self.recorder = recorder

    def forward(self, model, batch):
        ret = model(batch)
        return ret
