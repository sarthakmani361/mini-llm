"""utils.py -- small helpers shared by train.py and generate.py."""

import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # Apple Silicon GPUs
        return "mps"
    return "cpu"
