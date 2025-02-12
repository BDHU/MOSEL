import torch
import copy

def synchronize_gpus(devices=None):
    if devices is None:
        raise AssertionError("Must specify either devices (array)")
    
    if 'cpu' in devices:
        return
    
    for d in devices:
        device = torch.device("cuda:" + str(d))
        torch.cuda.synchronize(device=device)

def dict_to_half(dictionary):
    for k, v in dictionary.items():
        if torch.is_tensor(v):
            dictionary[k] = v.half()
    return dictionary 

def dict_to(dictionary, device=None):
    r"""
    Move all dict's value to gpu if the value is a tensor
    """
    for k, v in dictionary.items():
        if torch.is_tensor(v):
            if device != "cpu":
                cuda_dev = torch.device("cuda:" + str(device))
                dictionary[k] = v.to(cuda_dev)
            else:
                dictionary[k] = v.to(device=device)
        else:
            dictionary[k] = v
    return dictionary

def copy_dict_to(dictionary, device=None):
    r"""
    Move all dict's value to gpu if the value is a tensor
    """

    new_dict = {}
    for k, v in dictionary.items():
        if torch.is_tensor(v):
            if device != "cpu":
                cuda_dev = torch.device("cuda:" + str(device))
                new_dict[k] = v.to(device=cuda_dev, non_blocking=True, copy=True)
            else:
                new_dict[k] = v.to(device=device)
        else:
            new_dict[k] = v
    return new_dict

def dict_shared(dictionary):
    for k, v in dictionary.items():
        if torch.is_tensor(v):
            v.share_memory_()




def all_max_memories(devices=None):
    if devices is None:
        raise AssertionError("devices can not be None")

    return [torch.cuda.max_memory_allocated(device=torch.device("cuda:"+str(device))) \
            for device in devices]
