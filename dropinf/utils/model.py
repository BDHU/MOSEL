import torch
import copy

def model_to(model, device=None):
    r"""
    Move a model to a specified GPU
    """
    if device == "cpu":
        return model.to(device=device)
    cuda_dev = torch.device("cuda:" + str(device))
    return model.to(cuda_dev)


def copy_model_to_gpus(model=None, devices=None):
    r"""
    Copy a model to multiple GPUs
    """
    if devices is None:
        raise AssertionError("Must specify devices (array)")

    num_gpus = len(devices)
    models = [None] * num_gpus
    for i, d in enumerate(devices):
        device = torch.device("cuda:" + str(d))
        models[i] = copy.deepcopy(model).to(device=device)

    return models
