import torch
from torchmetrics import Metric

from functools import reduce
import statistics

# class Accuracy(Metric):
    
#     def __init__(self, dist_sync_on_step=False):
#         raise NotImplementedError("Please define accuracy init function")
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

#     def update(self, logits, target): 
#         raise NotImplementedError("Please define accuracy update function")
#         logits, target = (
#             torch.tensor(logits.detach().cpu().numpy()).to(self.correct.device),
#             torch.tensor(target.detach().cpu().numpy()).to(self.correct.device),
#         )
#         if logits.size(1) > 1:
#             preds = logits.argmax(dim=-1).squeeze()
#         else:
#             preds = (logits >= 0).squeeze()
#         target = target.squeeze()    
#         preds = preds.to(target.device)
#         preds = preds[target != -100]
#         target = target[target != -100]
#         if target.numel() == 0:
#             return 1

#         assert preds.shape == target.shape

#         self.correct += torch.sum(preds == target)
#         self.total += target.numel()

#     def compute(self):
#         raise NotImplementedError("Please define accuracy compute function")
#         return self.correct / self.total

#     def stdev(self):
#         return -1
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


class Accuracy(Metric):
    
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


class Memory(Metric):

    # full_state_update: bool = True
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("max_memory_used", default=[], dist_reduce_fx="cat")

    def update(self, **kwargs):
        if len(kwargs) == 1 and "mem" in kwargs \
        and isinstance(kwargs["mem"], int):
            self.max_memory_used.append(kwargs["mem"])
        else:
            raise NotImplementedError("Memory metric doesn't support the function yet")

    def compute(self):
        return sum(self.max_memory_used) / len(self.max_memory_used)

    def stdev(self):
        return statistics.stdev(self.max_memory_used)


class Latency(Metric):

    # full_state_update: bool = True

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("latency", default=[], dist_reduce_fx="cat")

    def update(self, **kwargs):
        if len(kwargs) == 1 and \
        "lat" in kwargs:
            self.latency.append(kwargs["lat"])
        else:
            raise NotImplementedError("Latency metric diesn't support the function yet")

    def compute(self):
        return sum(self.latency) / len(self.latency)

    def stdev(self):
        return statistics.stdev(self.latency)
