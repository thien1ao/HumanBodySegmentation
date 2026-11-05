# Human Body Mesh Segmentation with Spectral Features and DiffusionNet

Semantic segmentation of 3D human body meshes into eight anatomical parts: **head, hand, forearm, upper arm, torso, thigh, shin, foot**.

The network only sees **intrinsic** geometry: the Laplace–Beltrami eigenbasis and the Heat Kernel Signature (HKS). Because of this, its predictions do not depend on how the mesh is oriented or positioned in space.

<p align="center">
  <img src="assets/samples.png" alt="Ground-truth segmentations from the training set" height="600">
</p>

## Highlights

| Metric (SHREC'07 test set, 18 meshes) | Value     |
|---------------------------------------|-----------|
| Face accuracy                         | **88.1%** |
| Mean IoU                              | **79.9%** |

- Orientation-invariant input features (HKS at 16 log-spaced diffusion times)
- A small DiffusionNet-style model with learned per-channel spectral diffusion
- The full pipeline (preprocessing, training, evaluation, visualization) is in plain PyTorch

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Method](#method)
- [Results](#results)
- [References](#references)

## Repository Structure

```
HumanBodySegmentation/
├── python_scripts/
│   ├── data.py        # Mesh loading, normalization, edge-to-face labels, alignment
│   ├── features.py    # Laplace–Beltrami spectrum and Heat Kernel Signature
│   ├── model.py       # MiniDiffusionNet architecture
│   ├── train.py       # Training loop and checkpointing (entry point)
│   ├── evaluate.py    # Inference, confusion matrix, IoU
│   └── plotting.py    # Visualization utilities
├── notebooks/
│   └── main.ipynb     # End-to-end walkthrough: data, features, training, evaluation
├── assets/            # Figures used in this README
├── human_seg/         # Dataset (download separately, see below)
└── model.pt           # Trained checkpoint
```

## Installation

Python 3 with the following packages is required:

```bash
pip install torch numpy scipy matplotlib robust-laplacian
```

[PyTorch3D](https://github.com/facebookresearch/pytorch3d) is used for OBJ loading. Follow its
[installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for your CUDA/PyTorch version.

## Dataset

This project uses the preprocessed human segmentation benchmark from [MeshCNN](https://github.com/ranahanocka/MeshCNN)
([download](https://www.dropbox.com/scl/fi/r04ahap6vhuwzkjj9mpwk/human_seg.tar.gz?rlkey=dyjwx6ehxmarvtrzl6atvs9mz&e=1)). Extract it into the repository root:

```bash
tar -xzf human_seg.tar.gz   # creates human_seg/{train,test,seg,sseg}
```

| Split | Source             | Meshes |
|-------|--------------------|-------:|
| Train | MIT animation      | 169    |
| Train | FAUST              | 100    |
| Train | SCAPE              | 71     |
| Train | Adobe              | 41     |
| Test  | SHREC'07           | 18     |

Labels come as per-edge annotations. Each face gets the majority label of its three edges.

## Usage

**Train** a model from scratch (100 epochs; uses CUDA if available). The checkpoint is written to `model.pt`:

```bash
cd python_scripts
python train.py
```

**Explore and evaluate** with the notebook. It covers data inspection, eigenvector and HKS visualizations, training curves, per-mesh predictions, the confusion matrix, and per-class IoU:

```bash
jupyter notebook notebooks/main.ipynb
```

## Method

![Pipeline](assets/pipeline.png)

1. **Normalization.** Each mesh is centered on its area-weighted centroid and scaled to unit surface area.
2. **Spectral decomposition.** The 128 smallest eigenpairs of the robust Laplacian
   ([Sharp & Crane, 2020](https://github.com/nmwsharp/robust-laplacians-py)) are computed, and the constant eigenvector is dropped.
3. **Heat Kernel Signature.** HKS is evaluated at 16 log-spaced times, chosen from the median spectrum of the training set, then log-scaled and standardized with training-set statistics.
4. **MiniDiffusionNet.** A linear embedding (width 128) is followed by 4 residual diffusion blocks. Each block:
   - diffuses features in the spectral domain with a **learned per-channel diffusion time** *t*:
     `x̃ = Φ · diag(exp(−λ·t)) · Φᵀ M x`
   - concatenates `[x, x̃]` and applies a two-layer MLP with dropout as a residual update.

   Per-vertex logits are averaged over each face's vertices to give per-face predictions.
   See [`python_scripts/model.py`](python_scripts/model.py).
5. **Training.** Cross-entropy loss, AdamW (lr 1e-3, weight decay 1e-4), cosine annealing over 100 epochs, and Gaussian feature noise (σ = 0.05) for augmentation.

## Results

### Sample predictions

<p align="center">
  <img src="assets/predictions/shrec__1_prediction.png" alt="Prediction on SHREC'07 mesh 1" width="32%">
  <img src="assets/predictions/shrec__10_prediction.png" alt="Prediction on SHREC'07 mesh 10" width="32%">
  <img src="assets/predictions/shrec__11_prediction.png" alt="Prediction on SHREC'07 mesh 11" width="32%">
</p>

### Per-class IoU

| Head  | Hand  | Forearm | Upper arm | Torso | Thigh | Shin  | Foot  | **Mean** |
|:-----:|:-----:|:-------:|:---------:|:-----:|:-----:|:-----:|:-----:|:--------:|
| 75.8  | 84.7  | 84.0    | 69.6      | 82.3  | 75.7  | 82.5  | 84.8  | **79.9** |

### Training curves

![Training curves](assets/history.png)

Test accuracy levels off at about 88% after roughly 15 epochs, while training accuracy keeps climbing to 96.6%. This points to mild overfitting.

### Confusion matrix

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion matrix" width="400">
</p>

The matrix is row-normalized and area-weighted. Almost all errors fall at boundaries between parts that touch:
thigh → torso (hip), upper arm → torso (shoulder), foot → shin (ankle), hand → forearm (wrist), and head → torso (neck).

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG, 2022. [[code]](https://github.com/nmwsharp/diffusion-net)
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP, 2009.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes.* SGP, 2020. [[code]](https://github.com/nmwsharp/robust-laplacians-py)
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH, 2019. [[code]](https://github.com/ranahanocka/MeshCNN)
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH, 2017.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* arXiv, 2020. [[code]](https://github.com/facebookresearch/pytorch3d)
- Datasets: SCAPE (Anguelov et al., 2005), MIT animation (Vlasic et al., 2008), FAUST (Bogo et al., 2014), SHREC'07 (Giorgi et al., 2007).
