"""Create human_seg-style annotation files (.eseg / .seseg) for a new mesh from model predictions.

Usage:
    python python_scripts/annotate.py path/to/mesh.obj [--out DIR] [--up y]

Writes, next to the mesh (or into --out):
    <name>.eseg       one label per edge, 1..8, edge order as in data.get_ids_from_edges
    <name>.seseg      8 soft values per edge: 0.5 for the class of each adjacent face
    <name>_faces.txt  one label per face, 0..7
    <name>_seg.png    figure of the segmentation (front and back)

These are PREDICTED labels (pseudo-labels), not ground truth. Check the figure and fix
mistakes before using them to train or evaluate.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from data import NUM_CLASSES, CLASS_NAMES, compute_areas, get_ids_from_edges, get_face_labels_from_edges
from predict import load_checkpoint, segment_obj, to_y_up


def summarize(e, pred):
    area = compute_areas(e["V"], e["Fc"])
    share = torch.zeros(NUM_CLASSES, dtype=area.dtype).index_add_(0, pred, area) / area.sum()
    print(f"{e['name']}: {len(e['V'])} vertices, {len(e['Fc'])} faces")
    for name, s in zip(CLASS_NAMES, share):
        print(f"  {name:10s} {100 * s:5.1f}% of surface")


def face_to_edge_labels(Fc, face_labels):
    """Returns (eseg labels 1..C, seseg soft labels [E, C]) in get_ids_from_edges order."""
    edge_id = get_ids_from_edges(Fc)
    E = len(edge_id)
    first = np.full(E, -1)
    soft = np.zeros((E, NUM_CLASSES))
    for f, face in enumerate(Fc.tolist()):
        lab = int(face_labels[f])
        for k in range(3):
            e = edge_id[tuple(sorted((face[k], face[(k + 1) % 3])))]
            soft[e, lab] = 0.5
            if first[e] < 0:
                first[e] = lab
    return first + 1, soft


def fix_round_trip(Fc, face_labels, eseg):
    """Boundary edges can carry only one of their two labels. Flip ones that make the
    face-from-edges majority vote disagree with the intended face label."""
    edge_id = get_ids_from_edges(Fc)
    for _ in range(5):
        decoded = get_face_labels_from_edges(Fc, eseg).numpy()
        bad = np.nonzero(decoded != face_labels)[0]
        if len(bad) == 0:
            break
        for f in bad:
            face = Fc[f].tolist()
            for k in range(3):
                e = edge_id[tuple(sorted((face[k], face[(k + 1) % 3])))]
                eseg[e] = face_labels[f] + 1
    return eseg


def plot_views(V, Fc, labels, title, filename):
    import plotting
    fig = plt.figure(figsize=(10, 6))
    for i, azim in enumerate([40, 220]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        plotting.draw_mesh(ax, {"V": V, "Fc": Fc, "labels": labels}, azim=azim)
        ax.set_title("front" if i == 0 else "back")
    fig.suptitle(title)
    plotting.class_legend(fig)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("obj")
    parser.add_argument("--out", help="output directory (default: next to the mesh)")
    parser.add_argument("--up", choices=["x", "y", "z"], default="y", help="up axis, for the figure only")
    args = parser.parse_args()

    obj = Path(args.obj)
    out = Path(args.out) if args.out else obj.parent
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, t, mean, std = load_checkpoint(device)
    e, pred = segment_obj(obj, model, t, mean, std, device)
    summarize(e, pred)

    face_labels = pred.numpy()
    eseg, seseg = face_to_edge_labels(e["Fc"], face_labels)
    eseg = fix_round_trip(e["Fc"], face_labels, eseg)
    decoded = get_face_labels_from_edges(e["Fc"], eseg).numpy()
    print(f"round trip faces -> .eseg -> faces: {(decoded == face_labels).mean():.4f} agree")

    name = obj.stem
    np.savetxt(out / f"{name}.eseg", eseg, fmt="%d")
    np.savetxt(out / f"{name}.seseg", seseg, fmt="%.6f")
    np.savetxt(out / f"{name}_faces.txt", face_labels, fmt="%d")
    plot_views(to_y_up(e["V"], args.up), e["Fc"], pred, f"{name} (predicted)", out / f"{name}_seg.png")
    for suffix in [".eseg", ".seseg", "_faces.txt", "_seg.png"]:
        print("wrote", out / f"{name}{suffix}")
