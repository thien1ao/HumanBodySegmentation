# Human Body Segmentation

Semantic segmentation of 3D human meshes: given a triangle mesh of a person, predict for every triangle which
body part it belongs to — **head, hand, forearm, upper arm, torso, thigh, shin, foot**.

## Project overview

This project is inspired by **[MeshCNN](https://ranahanocka.github.io/MeshCNN/)**
(Hanocka et al., *"MeshCNN: A Network with an Edge"*, SIGGRAPH 2019), which showed that neural networks can
operate directly on triangle meshes, and uses the human body segmentation dataset released with it.

Instead of MeshCNN's edge convolutions and pooling, this project takes a spectral, intrinsic approach:

1. **Intrinsic features.** The meshes come from different sources and are oriented inconsistently, so raw
   `(x, y, z)` coordinates don't generalize. Each vertex is described by the **Heat Kernel Signature (HKS)**,
   computed from the eigenfunctions of the cotangent Laplace–Beltrami operator. It is invariant to rotation and
   nearly invariant to pose.
2. **Learned heat diffusion.** A compact network in the spirit of
   **[DiffusionNet](https://github.com/nmwsharp/diffusion-net)** (Sharp et al., 2022) propagates features
   across the surface with per-channel learnable diffusion times, so a single layer can share information
   across the whole body.
3. **Per-face prediction.** Vertex scores are averaged over each triangle's three corners.

Everything is in [`human_segmentation.ipynb`](human_segmentation.ipynb), which explains each step in a markdown
cell: loading and label conversion, rendering with PyTorch3D, Laplacian eigenfunctions, HKS, the model,
training, evaluation and visualization of predictions.

### Results (test set, 18 SHREC meshes)

| Metric | Value |
|---|---|
| Face accuracy | 84.5% |
| Area-weighted accuracy | 85.1% |
| Mean IoU | 0.733 |

Training reaches 95.5% accuracy, so the model overfits somewhat. Most errors lie on part boundaries; the weakest
classes are head and upper arm. The test set was also used to monitor training (following the MeshCNN protocol),
so these numbers are slightly optimistic.

## Dataset overview

The `human_seg` dataset from MeshCNN collects human meshes from five public sources. Every mesh was simplified to
about **750 vertices / 1500 triangles**.

| Split | Source | Meshes | Description |
|---|---|---|---|
| train | MIT animation | 169 | Frames of captured human motion sequences (Vlasic et al.) |
| train | FAUST | 100 | Scans of 10 people in 10 poses each |
| train | SCAPE | 71 | One person in 71 poses |
| train | Adobe Fuse | 41 | Synthetic characters, including stylized/cartoon bodies, mostly in T-pose |
| test | SHREC | 18 | Meshes from the SHREC benchmark — a source not present in training |
| | **Total** | **399** | 381 train / 18 test |

Many training meshes are different poses of the same few people, so the number of distinct bodies is much
smaller than 381. Orientation also differs between sources (the "up" axis is y for FAUST, x for SCAPE, mixed for
the others), and the test set is oriented differently from all of the training data.

### Folder layout

```
human_seg/
├── train/   *.obj     training meshes
├── test/    *.obj     test meshes
├── seg/     *.eseg    hard labels: one integer (1–8) per edge
└── sseg/    *.seseg   soft labels: 8 probabilities per edge (used for MeshCNN's evaluation)
```

Labels are stored **per edge**, in MeshCNN's edge order: walk the faces in file order, visit each face's edges
`(a,b)`, `(b,c)`, `(c,a)`, and number each edge the first time it appears. The notebook rebuilds this order and
converts edge labels to face labels by majority vote over each triangle's three edges.

| Label | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Part | head | hand | forearm | upper arm | torso | thigh | shin | foot |

The dataset is distributed with MeshCNN (see `scripts/human_seg/get_data.sh` in the
[MeshCNN repository](https://github.com/ranahanocka/MeshCNN)). The original meshes remain under the licenses of
their respective sources.

## Getting started

Requirements: Python 3.10, PyTorch, [PyTorch3D](https://github.com/facebookresearch/pytorch3d), NumPy, SciPy,
matplotlib, Jupyter. A GPU is optional; training takes about a minute on a GPU.

1. Download the dataset and place it in `human_seg/` at the repository root.
2. Open `human_segmentation.ipynb` and run all cells.
3. The trained model and its preprocessing constants are saved to `checkpoints/human_seg_diffusionnet.pt`.

## References

- R. Hanocka, A. Hertz, N. Fish, R. Giryes, S. Fleishman, D. Cohen-Or. *MeshCNN: A Network with an Edge.*
  ACM Transactions on Graphics (SIGGRAPH), 2019.
- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.*
  ACM Transactions on Graphics, 2022.
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat
  Diffusion.* Symposium on Geometry Processing, 2009.
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH 2017
  (compiled the human segmentation dataset later used by MeshCNN).
