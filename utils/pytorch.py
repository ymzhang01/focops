import torch
import numpy as np

def graph_detach(*args):
    return [arg.detach() for arg in args]


def to_dytype_device(dtype, device, *args):
    return [arg.to(dtype).to(device) for arg in args]

def to_device(device, *args):
    return [arg.to(device) for arg in args]

def torch_to_numpy(*args):
    if len(args) == 1:
        return args[0].detach().cpu().numpy()
    else:
        return [arg.detach().cpu().numpy() for arg in args]

