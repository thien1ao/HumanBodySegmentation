import os
import glob
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_obj

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "human_seg"
ASSETS = ROOT / "assets"
MODEL_PATH = ROOT / "model.pt"

CLASS_NAMES = ["head", "hand", "forearm", "upper arm", "torso", "thigh", "shin", "foot"]
NUM_CLASSES = len(CLASS_NAMES)


def source(p):
    return os.path.basename(p).split("__")[0]


def list_meshes(path=DATA_PATH):
    train_paths = sorted(glob.glob(f"{path}/train/*.obj"))
    test_paths = sorted(glob.glob(f"{path}/test/*.obj"))
    assert train_paths, f"no training meshes found in {path}/train"
    assert test_paths, f"no test meshes found in {path}/test"
    return train_paths, test_paths


def get_ids_from_edges(faces):
    edge_ids = {}
    for face in faces.tolist():
        for e in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            e = tuple(sorted(e))
            if e not in edge_ids:
                edge_ids[e] = len(edge_ids)
    return edge_ids


def get_face_labels_from_edges(faces, edge_labels):
    edge_id = get_ids_from_edges(faces)
    assert len(edge_id) == len(edge_labels)
    face_labels = []

    for face in faces.tolist():
        labels = []
        for k in range(3):
            edge = tuple(sorted((face[k], face[(k + 1) % 3])))
            labels.append(edge_labels[edge_id[edge]])

        if labels[0] == labels[1] or labels[0] == labels[2]:
            label = labels[0]
        elif labels[1] == labels[2]:
            label = labels[1]
        else:
            label = labels[0]
        face_labels.append(label)

    return torch.tensor(face_labels) - 1


def compute_areas(V, faces):
    e1 = V[faces[:, 1]] - V[faces[:, 0]]
    e2 = V[faces[:, 2]] - V[faces[:, 0]]
    return 0.5 * torch.linalg.cross(e1, e2).norm(dim=1)


def load_mesh(obj_path):
    verts, faces, _ = load_obj(str(obj_path), load_textures=False)
    V, Fc = verts.double(), faces.verts_idx.long()

    # Center on the area-weighted centroid and normalize to unit surface area
    A = compute_areas(V, Fc)
    face_centers = V[Fc].mean(1)
    centroid = (face_centers * A[:, None]).sum(0) / A.sum()
    V = (V - centroid) / torch.sqrt(A.sum())

    name = os.path.splitext(os.path.basename(obj_path))[0]
    return {"name": name, "V": V, "Fc": Fc}


def load_example(obj_path, path=DATA_PATH):
    e = load_mesh(obj_path)
    edge_labels = np.loadtxt(f"{path}/seg/{e['name']}.eseg", dtype=np.int64)
    e["labels"] = get_face_labels_from_edges(e["Fc"], edge_labels)
    return e


def bring_to_same_frame(ex):
    V, Fc, labels = ex["V"], ex["Fc"], ex["labels"]
    centers = V[Fc].mean(1)
    part = lambda i: centers[labels == i].mean(0)

    head, shin, foot = part(0), part(6), part(7)
    y = head - foot
    y = y / y.norm()  # up

    z = foot - shin
    z = z - (z @ y) * y
    z = z / z.norm()  # forward

    x = torch.linalg.cross(y, z)
    R = torch.stack([x, y, z], 1)
    return {**ex, "V": V @ R}
