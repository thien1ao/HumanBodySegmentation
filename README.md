# 3D Human Mesh Part Segmentation using DiffusionNet

## Overview
A deep learning pipeline for semantic 3D human mesh segmentation (8 classes) that leverages intrinsic geometric features (Laplace–Beltrami and HKS) to achieve spatial invariance, ensuring robust generalization across diverse poses and datasets.

![Sample meshes with ground-truth body-part labels](assets/samples.png)


## Data

We use the human body segmentation benchmark introduced by Maron et al. (2017), in the preprocessed version distributed with MeshCNN (Hanocka et al., 2019). It is stored in `human_seg/`:

```
human_seg/
├── train/   381 meshes (.obj)
├── test/     18 meshes (.obj)
├── seg/     hard labels, one class per edge (.eseg)
└── sseg/    soft labels, 8 class probabilities per edge (.seseg)
```

### Sources

The training and test sets come from different datasets, so the test set measures generalization to unseen subjects and modeling styles.

| Split | Source | Meshes | Description |
|-------|--------|-------:|-------------|
| train | **MIT animation** (Vlasic et al., 2008) | 169 | Frames of captured motion sequences; many poses of a few subjects |
| train | **FAUST** (Bogo et al., 2014) | 100 | Registered 3D scans of 10 people in 10 poses |
| train | **SCAPE** (Anguelov et al., 2005) | 71 | One subject in many poses |
| train | **Adobe Fuse** | 41 | Synthetic characters with varied body shapes |
| test  | **SHREC'07** (Giorgi et al., 2007) | 18 | Human models from the SHREC'07 watertight track |

Each source uses its own coordinate frame, so the orientation of the body is not consistent across the dataset.

### Mesh format

All meshes were remeshed to the same resolution:

- **752 vertices, 1500 triangles, 2250 edges**
- closed, genus-0 surfaces (`V − E + F = 2`)

### Labels

The labels are given **per edge**: each `.eseg` file has 2250 lines with a class index from 1 to 8. We convert them to **per-face** labels by majority vote over the three edges of each triangle (if all three disagree, the first edge wins), then shift them to the range 0–7.

Class distribution over all edges in the dataset:

| Class | Body part | Share |
|------:|-----------|------:|
| 0 | head      |  7.1 % |
| 1 | hand      |  4.9 % |
| 2 | forearm   |  7.3 % |
| 3 | upper arm |  8.7 % |
| 4 | torso     | 31.3 % |
| 5 | thigh     | 16.2 % |
| 6 | shin      | 15.5 % |
| 7 | foot      |  9.1 % |

The classes are imbalanced: the torso covers almost a third of the surface, while hands cover less than 5 %. The labels do not separate left from right ("hand" covers both hands).

## Methodology

### 1. Preprocessing

For every mesh:

1. Load vertices `V` and faces `F` (`pytorch3d.io.load_obj`).
2. Convert edge labels to face labels as described above.
3. **Center** the mesh at its area-weighted centroid.
4. **Normalize the scale** so that the total surface area is 1:

$$
V \leftarrow \frac{V}{\sqrt{A_{\text{total}}}}
$$

With equal areas, the spectra of different meshes can be compared directly, and the same diffusion times can be used for all of them (see below).

### 2. Laplace–Beltrami operator

The Laplace–Beltrami operator (LBO) generalizes the Laplacian to curved surfaces. On a triangle mesh we discretize it with the **cotangent Laplacian** `L` and the **lumped mass matrix** `M`:

$$
L_{ij} =
\begin{cases}
-\tfrac{1}{2}\left(\cot \alpha_{ij} + \cot \beta_{ij}\right) & \text{if } (i,j) \text{ is an edge} \\
-\sum_{k \neq i} L_{ik} & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}
\qquad
M_{ii} = \tfrac{1}{3} \sum_{f \ni i} A_f
$$

where `α_ij` and `β_ij` are the two angles opposite edge `(i, j)`, and `A_f` is the area of face `f`. In practice we compute `L` and `M` with `robust_laplacian`, which also handles degenerate triangles.

We then solve the generalized eigenvalue problem

$$
L \, \phi_k = \lambda_k \, M \, \phi_k, \qquad 0 = \lambda_0 \le \lambda_1 \le \dots \le \lambda_{K-1}
$$

for the first `K = 128` eigenpairs (`scipy.sparse.linalg.eigsh` with shift-invert). The eigenfunctions `φ_k` are the surface analogue of Fourier modes: low `k` gives smooth, body-scale variations, and high `k` gives fine detail. The constant eigenfunction `φ_0` carries no information and is dropped.

The LBO depends only on lengths and angles measured **along the surface**. As a result, it is:

- **invariant to rotation and translation**, so the different coordinate frames of the sources no longer matter;
- **approximately invariant to pose**, because bending a limb barely stretches the surface (the deformation is close to an isometry);
- **invariant to scale**, after the area normalization above.

### 3. Heat Kernel Signature

The eigenfunctions themselves cannot be used directly as features: their sign is arbitrary, and eigenfunctions with similar eigenvalues can swap order between meshes. Instead we use the **Heat Kernel Signature** (Sun et al., 2009). It measures how much heat stays at a point `x` a time `t` after a unit of heat was placed there:

$$
\mathrm{HKS}(x, t) = \sum_{k} e^{-\lambda_k t} \, \phi_k(x)^2
$$

Because it uses `φ_k²`, HKS is not affected by the sign ambiguity. Small `t` describes the local shape around `x` (fingertips, toes and the head stand out), and large `t` describes the position of `x` within the whole body.

We evaluate HKS at **16 fixed times**, log-spaced in `[0.01, 1]`. The same times are used for every mesh, which is possible because all meshes have unit area. Each time scale is normalized by the heat trace and log-transformed to compress its dynamic range:

$$
\widetilde{\mathrm{HKS}}(x, t) = \log\left(\frac{\mathrm{HKS}(x, t)}{\sum_k e^{-\lambda_k t}} + \varepsilon\right)
$$

```python
def hks(evals, evecs, times):
    # evals (k,), evecs (n, k), times (T,) -> (n, T)
    decay = torch.exp(-evals[:, None] * times[None, :])   # (k, T)
    h = (evecs ** 2) @ decay                               # (n, T)
    return torch.log(h / decay.sum(0, keepdim=True) + 1e-12)
```

### 4. Per-face features

HKS is defined per vertex. We average it over the three vertices of each triangle, which gives a **1500 × 16** feature matrix per mesh that matches the per-face labels. Each feature channel is standardized with the mean and standard deviation of the training set. The features are computed once and cached, since the eigendecomposition only depends on the mesh.

## References

- D. Anguelov et al. *SCAPE: Shape Completion and Animation of People.* SIGGRAPH 2005.
- D. Vlasic et al. *Articulated Mesh Animation from Multi-view Silhouettes.* SIGGRAPH 2008.
- D. Giorgi, S. Biasotti, L. Paraboschi. *SHREC'07: Watertight Models Track.* 2007.
- F. Bogo et al. *FAUST: Dataset and Evaluation for 3D Mesh Registration.* CVPR 2014.
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP 2009.
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH 2017.
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH 2019.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes* (robust_laplacian). SGP 2020.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* arXiv:2007.08501, 2020.

### Software

- **PyTorch3D** (mesh loading): https://github.com/facebookresearch/pytorch3d
- **robust_laplacian** (cotangent Laplacian and mass matrix): https://github.com/nmwsharp/robust-laplacians-py
- **SciPy** (sparse eigensolver `eigsh`): https://github.com/scipy/scipy
- **MeshCNN** (source of the preprocessed `human_seg` dataset): https://github.com/ranahanocka/MeshCNN
