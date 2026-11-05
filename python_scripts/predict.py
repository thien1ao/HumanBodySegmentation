"""
Usage:
    python python_scripts/predict.py path/to/mesh.obj
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from data import MODEL_PATH, NUM_CLASSES, CLASS_NAMES, compute_areas, list_meshes, load_mesh
from features import add_spectrum, hks, hks_stats, hks_times, add_hks, preprocess, to_device
from model import MiniDiffusionNet
from evaluate import predict


def load_checkpoint(device, path=MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu")
    if "t" not in ckpt:
        train_data = preprocess(list_meshes()[0])
        t = hks_times(train_data)
        add_hks(train_data, t)
        mean, std = hks_stats(train_data)
        ckpt.update(t=t, mean=mean, std=std)
        torch.save(ckpt, path)
        print(f"added t/mean/std to {path}")

    t, mean, std = ckpt["t"], ckpt["mean"], ckpt["std"]
    model = MiniDiffusionNet(len(t), NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, t, mean, std


def segment_obj(obj_path, model, t, mean, std, device, k=128):
    e = add_spectrum(load_mesh(obj_path), k)
    if (e["evals"][:3] < 1e-6).sum() > 0:
        print("warning: several ~zero eigenvalues, the mesh probably has disconnected pieces")
    e["hks"] = hks(e["evals"], e["evecs"], t)
    pred = predict(model, to_device(e, mean, std, device))
    return e, pred


def to_y_up(V, up="y"):
    """Rotate so `up` points along +y (only for drawing)."""
    if up == "z":
        return V[:, [0, 2, 1]] * torch.tensor([1, 1, -1], dtype=V.dtype)
    if up == "x":
        return V[:, [1, 0, 2]] * torch.tensor([-1, 1, 1], dtype=V.dtype)
    return V


if __name__ == "__main__":
    import plotting

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("obj", help="path to the .obj file")
    parser.add_argument("--up", choices=["x", "y", "z"], default="y", help="up axis of the mesh, for drawing")
    parser.add_argument("--save", help="save the figure to this path")
    parser.add_argument("--labels", help="write per-face labels (0-7, one per line) to this file")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, t, mean, std = load_checkpoint(device)
    e, pred = segment_obj(args.obj, model, t, mean, std, device)

    if args.labels:
        np.savetxt(args.labels, pred.numpy(), fmt="%d")
        print(f"wrote {args.labels}")

    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection="3d")
    plotting.draw_mesh(ax, {"V": to_y_up(e["V"], args.up), "Fc": e["Fc"], "labels": pred})
    ax.set_title(f"{e['name']} (prediction)")
    plotting.class_legend(fig)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plotting._save_and_show(args.save)
