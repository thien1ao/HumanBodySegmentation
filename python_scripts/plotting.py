import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import Patch
from pathlib import Path

from data import ASSETS, CLASS_NAMES, NUM_CLASSES, source, load_example, bring_to_same_frame
from evaluate import predict

COLORS = torch.tensor([
    [0.80, 0.30, 0.30],
    [0.30, 0.70, 0.30],
    [0.30, 0.45, 0.90],
    [0.95, 0.75, 0.20],
    [0.70, 0.35, 0.85],
    [0.20, 0.80, 0.80],
    [0.95, 0.55, 0.15],
    [0.60, 0.60, 0.60],
])

ERROR_COLORS = torch.tensor([
    [0.85, 0.85, 0.85],
    [0.90, 0.10, 0.10],
])


def _set_equal_axes(ax, V, elev=10, azim=40):
    mins, maxs = V.min(axis=0), V.max(axis=0)
    center, radius = (mins + maxs) / 2, (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, vertical_axis="y")


def _save_and_show(filename):
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def draw_mesh(ax, ex, colors=COLORS, elev=10, azim=40):
    V = ex["V"].detach().cpu().numpy()
    Fc = ex["Fc"].detach().cpu().numpy()
    C = ex["labels"].detach().cpu().numpy()
    face_colors = colors.detach().cpu().numpy()[C]

    mesh = art3d.Poly3DCollection(V[Fc], facecolors=face_colors, edgecolor="black", linewidth=0.1)
    ax.add_collection3d(mesh)
    _set_equal_axes(ax, V, elev, azim)


def draw_scalar(ax, V, Fc, values, cmap, norm, elev=10, azim=40):
    face_vals = values[Fc].mean(axis=1)
    mesh = art3d.Poly3DCollection(V[Fc], facecolors=cmap(norm(face_vals)), edgecolor="none")
    ax.add_collection3d(mesh)
    _set_equal_axes(ax, V, elev, azim)


def class_legend(fig, colors=COLORS):
    palette = colors.detach().cpu().numpy()
    handles = [Patch(facecolor=palette[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=NUM_CLASSES, fontsize=9, frameon=False)


def plot_sample(ex, colors=COLORS, figsize=(6, 6), elev=10, azim=40):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, ex, colors, elev, azim)
    plt.show()


def plot_samples(train_paths, test_paths, colors=COLORS, ncols=4, size=4, elev=10, azim=40, filename=ASSETS / "samples.png"):
    picks = {}
    for p in train_paths + test_paths:
        picks.setdefault(source(p), p)

    nrows = math.ceil(len(picks) / ncols)
    fig = plt.figure(figsize=(size * ncols, size * nrows + 1))
    for i, (src, obj_path) in enumerate(picks.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        draw_mesh(ax, bring_to_same_frame(load_example(obj_path)), colors, elev, azim)
        ax.set_title(src, fontsize=10)

    class_legend(fig, colors)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    _save_and_show(filename)


def plot_eigenvectors(V, Fc, evecs, evals, ks, filename=None):
    V, Fc = np.asarray(V), np.asarray(Fc)
    evecs = torch.as_tensor(evecs).detach().cpu().numpy()
    evals = torch.as_tensor(evals).detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 6))
    for i, k in enumerate(ks):
        ax = fig.add_subplot(2, math.ceil(len(ks) / 2), i + 1, projection="3d")
        f = evecs[:, k]
        m = np.abs(f[Fc].mean(axis=1)).max()
        draw_scalar(ax, V, Fc, f, plt.cm.RdBu_r, plt.Normalize(-m, m))
        ax.set_title(f"$\\phi_{{{k+1}}}$, $\\lambda$={evals[k]:.2f}")

    plt.tight_layout()
    _save_and_show(filename)


def plot_hks(V, Fc, H, t, cmap="magma", filename=ASSETS / "hks.png"):
    V, Fc = np.asarray(V), np.asarray(Fc)
    H = torch.as_tensor(H).detach().cpu().numpy()
    t = torch.as_tensor(t).detach().cpu().numpy()

    ks = np.linspace(0, len(t) - 1, 8, dtype=int)
    ncols = 4
    nrows = math.ceil(len(ks) / ncols)

    face_hks = np.stack([H[Fc, k].mean(axis=1) for k in ks])
    norm = plt.Normalize(face_hks.min(), face_hks.max())
    color_map = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(12, 3 * nrows))
    for i, k in enumerate(ks):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        draw_scalar(ax, V, Fc, H[:, k], color_map, norm)
        ax.set_title(f"HKS, t = {t[k]:.4f}")

    plt.tight_layout()
    _save_and_show(filename)


def plot_history(history, filename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel in [(ax1, "loss", "cross-entropy"), (ax2, "acc", "accuracy")]:
        ax.plot(history["epoch"], history[f"train_{key}"], label="train")
        ax.plot(history["test_epoch"], history[f"test_{key}"], "o-", label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(key)
        ax.legend()
    plt.tight_layout()
    _save_and_show(filename)


def plot_prediction(d, pred, filename=None):
    y = d["labels"]
    acc = (pred == y).float().mean().item()
    panels = [
        ("ground truth", y, COLORS),
        ("prediction", pred, COLORS),
        ("errors", (pred != y).long(), ERROR_COLORS),
    ]
    fig = plt.figure(figsize=(12, 4))
    for i, (title, labels, palette) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        draw_mesh(ax, {**d, "labels": labels}, palette)
        ax.set_title(title)
    fig.suptitle(f"{d['name']}   accuracy {acc:.3f}")
    plt.tight_layout()
    _save_and_show(filename)


def plot_predictions(model, data, data_dev, n=None, names=None, save_dir=None):

    pairs = list(zip(data, data_dev))

    if names is not None:
        pairs = [(d, dd) for d, dd in pairs if d["name"] in names]

    if n is not None:
        pairs = pairs[:n]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for d, dd in pairs:
        pred = predict(model, dd)

        filename = None
        if save_dir is not None:
            filename = save_dir / f"{d['name']}_prediction.png"

        plot_prediction(d, pred, filename=filename)


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, title="Confusion matrix", filename=ASSETS / "confusion_matrix.png"):
    cm_norm = (cm / cm.sum(1, keepdim=True).clamp(min=1e-12)).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    n = len(class_names)
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if i == j or v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.5 else "#333333")

    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    plt.tight_layout()
    _save_and_show(filename)
    return cm_norm
