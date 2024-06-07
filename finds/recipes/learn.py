"""Miscellaneous utilities

Copyright 2022, Terence Lim

MIT License
"""
from typing import List
import numpy as np
import random
from sklearn.model_selection import train_test_split
from pandas.api import types


def form_input(docs: List[List[int]], pad: int | None = 0) -> List[List[int]]:
    """Pad lists of index lists to form batch of equal lengths

    Args:
        docs: Input documents as lists of int lists
        pad: Value to pad with (None to pad with random value from list)

    Returns:
        List of padded lists of ints
    """
    lengths = [len(doc) for doc in docs]   # length of each doc
    max_length = max(lengths)              # to pad so all lengths equal max
    if max_length:                         # pad to max length
        out = [[0] * max_length if not not n else
               [doc + ([pad] * (max_length-n) if pad is not None else
                       random.choices(doc, k=max_length-n))]
               for doc, n in zip(docs, lengths)]
    else:   # all lines are blank
        out = [[0]] * len(lengths)
    return out


def form_batches(batch_size: int, idx: List) -> List[List[int]]:
    """Shuffles idx list into minibatches each of size batch_size

    Args:
        batch_size: Size of each minibatch
        idx: List of indexes

    Returns:
        List of batches of shuffled indexes 
    """
    idxs = [i for i in idx]
    random.shuffle(idxs)
    return [idxs[i:(i+batch_size)] for i in range(0, len(idxs), batch_size)]


def form_splits(labels: List[str | int] | int,
                test_size: float | int = 0.2,
                random_state: int = 42) -> List[List]:
    """Wraps over train_test_split to stratifies labels into split indexes

    Args:
        labels: Labels of series to stratify, or length of series to shuffle
        test_size: Desired size of test set as fraction or number of samples
        random_state: Set random seed

    Returns:
        tuple of stratified train indexes and test indexes
    """
    if types.is_list_like(labels):
        return train_test_split(np.arange(len(labels)),
                                stratify=labels,
                                random_state=random_state,
                                test_size=test_size)
    else:   # labels is an int
        return train_test_split(labels,
                                random_state=random_state,
                                test_size=test_size)


def torch_trainable(model, total: bool = True) -> List[int] | int:
    """Returns total number of trainable parameters in torch model"""
    import torch
    p = [p.numel() for p in model.parameters() if p.requires_grad]
    return sum(p) if total else p    # by components or total sum


def cuda_summary():
    """Print details of cuda environment

    Notes:

    - https://pytorch.org/get-started/locally/
    - check cuda version (e.g. 11.4?): Nvidia-smi
    - install matching torch version
    - pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    - you can specify $ export TORCH\_CUDA\_ARCH\_LIST="8.6" 
      in your environment to force it to build with SM 8.6 support
    """
    import torch
    print('version', torch.__version__)
    print('device capability', torch.cuda.get_device_capability())
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    print(torch.cuda.get_arch_list(), '(sm86 for rtx3080)')
    
def torch_summary(model):
    """Print torch network summary

    https://stackoverflow.com/questions/42480111/how-do-i-print-the-model-summary-in-pytorch
    """
    import torch
    modules = [module for module in model.modules()]
    params = [p.shape for p in model.parameters()]
    trains = [p.shape for p in model.parameters() if p.requires_grad]
    
    print(modules[0])
    total_params = 0
    total_train = 0
    for i in range(1,len(modules)):
        j = 2*i
        param = (params[j-2][1] * params[j-2][0]) + params[j-1][0]
        total_params += param
        train = (trains[j-2][1] * trains[j-2][0]) + trains[j-1][0]
        total_train += train
        print("Layer",i,"->\t",end="")
        print("Weights:", params[j-2][0],
              "x", params[j-2][1],
              "\tBias: ", params[j-1][0],
              "\tParameters: ", param,
              "\tTrainable: ", train)
    print("\nTotal Params: ", total_params,
          "\t\tTotal Trainable: ", total_train)
