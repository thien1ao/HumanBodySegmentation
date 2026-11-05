import math

import numpy as np
import torch
import robust_laplacian
from scipy.sparse.linalg import eigsh

from data import load_example, bring_to_same_frame


def add_spectrum(e, k=128):
    L, M = robust_laplacian.mesh_laplacian(e["V"].numpy(), e["Fc"].numpy())
    evals, evecs = eigsh(L, k=k, M=M, sigma=-1e-8)
    e["evals"] = torch.from_numpy(evals).clamp(min=0)[1:]
    e["evecs"] = torch.from_numpy(evecs)[:, 1:]
    e["mass"] = torch.from_numpy(M.diagonal().copy()).double()
    return e


def preprocess(paths, k=128):
    return [add_spectrum(bring_to_same_frame(load_example(p)), k) for p in paths]


def hks_times(train_data, n_times=16):
    lam1 = np.median([d["evals"][1].item() for d in train_data])
    lamK = np.median([d["evals"][-1].item() for d in train_data])
    return torch.logspace(math.log10(4 * math.log(10) / lamK), math.log10(4 * math.log(10) / lam1), n_times, dtype=torch.float64)


def hks(evals, evecs, t):
    decay = torch.exp(-evals[:, None] * t[None, :])
    h = (evecs ** 2) @ decay
    return torch.log(h / decay.sum(0, keepdim=True) + 1e-12)


def add_hks(data, t):
    for d in data:
        d["hks"] = hks(d["evals"], d["evecs"], t)


def hks_stats(train_data):
    train_hks = torch.cat([d["hks"] for d in train_data])
    return train_hks.mean(0), train_hks.std(0)


def to_device(d, mean, std, device):
    return {
        "x": ((d["hks"] - mean) / std).float().to(device),
        "evals": d["evals"].float().to(device),
        "evecs": d["evecs"].float().to(device),
        "mass": d["mass"].float().to(device),
        "Fc": d["Fc"].to(device),
        "y": d["labels"].to(device) if "labels" in d else None,
    }


def build_datasets(train_paths, test_paths, device, k=128, n_times=16):
    train_data = preprocess(train_paths, k)
    test_data = preprocess(test_paths, k)

    t = hks_times(train_data, n_times)
    add_hks(train_data + test_data, t)
    mean, std = hks_stats(train_data)

    train_dev = [to_device(d, mean, std, device) for d in train_data]
    test_dev = [to_device(d, mean, std, device) for d in test_data]
    return train_data, test_data, train_dev, test_dev, t
