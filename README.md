# Human Mesh Segmentation using DiffusionNet

## Overview
A deep learning pipeline for semantic human mesh segmentation (8 classes) that leverages intrinsic geometric features to achieve spatial invariance and robust generalization across different mesh poses and orientations.

![Sample meshes with ground-truth body-part labels](assets/samples.png)

## Dataset

The preprocessed human segmentation benchmark from [MeshCNN](https://github.com/ranahanocka/MeshCNN) ([download](https://www.dropbox.com/scl/fi/r04ahap6vhuwzkjj9mpwk/human_seg.tar.gz?rlkey=dyjwx6ehxmarvtrzl6atvs9mz&e=1)), extracted to `human_seg/`.

| Split | Source | Meshes |
|-------|--------|-------:|
| train | MIT animation (Vlasic et al., 2008) | 169 |
| train | FAUST (Bogo et al., 2014) | 100 |
| train | SCAPE (Anguelov et al., 2005) | 71 |
| train | Adobe Fuse | 41 |
| test  | SHREC'07 (Giorgi et al., 2007) | 18 |

## Methodology

### Normalization

Each mesh is centered at its area-weighted centroid and scaled to **unit surface area** ($`V \leftarrow V / \sqrt{A}`$). This makes eigenvalues comparable across meshes, so the same diffusion times can be used for all of them.

### Laplace–Beltrami Operator

While the standard Laplacian $\Delta f$ measures local deviation from neighboring values on a flat grid, the **Laplace–Beltrami operator** extends this concept to arbitrary curved surfaces. On discrete triangle meshes, we use the standard cotangent discretization:

$$(Lf)_i = \sum_{j \in N(i)} w_{ij}(f_i - f_j), \quad w_{ij} = \frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$$

where $\alpha_{ij}$ and $\beta_{ij}$ are the angles opposite edge $(i,j)$. To complete the operator, a lumped mass matrix $M$ is constructed, assigning each vertex a weight equal to one-third of the combined area of its adjacent triangles.

Because this formulation depends strictly on local angles and areas, $L$ is inherently invariant to rigid transformations (rotation and translation) and highly robust to isometric deformations such as body bending, which preserve surface metrics.

We construct $L$ and $M$ using [`robust_laplacian`](https://github.com/nmwsharp/robust-laplacians-py), which guarantees numerical stability even when handling degenerate triangles.

### Eigenfunctions

Solving the generalized eigenproblem:

$$L \phi_k = \lambda_k M \phi_k, \quad 0 = \lambda_0 \le \lambda_1 \le \dots$$

yields the mesh's spectral basis—acting as a generalized Fourier basis for non-Euclidean surfaces. Low eigenvalues capture global, large-scale structural patterns, while higher eigenvalues encode fine-grained geometric details.

We compute the first **128 non-trivial eigenpairs** using `scipy.sparse.linalg.eigsh` in shift-invert mode, discarding the constant zero-eigenfunction.

![Laplace–Beltrami eigenfunctions on a body mesh](assets/laplace.png)

### Heat Kernel Signature

Using the precomputed Laplace–Beltrami eigenbasis, the heat kernel admits a closed-form spectral expansion, simplifying the HKS to:

$$\text{HKS}(x, t) = k_t(x, x) = \sum_k e^{-\lambda_k t} \phi_k(x)^2$$

Because the formulation relies on squared eigenfunctions ($\phi_k^2$), it is naturally invariant to the sign ambiguity inherent in eigenvalue decomposition.

We evaluate the signature across **16 logarithmically spaced time steps** $t \in [0.01, 1]$. Since all meshes are normalized to unit surface area, each time channel corresponds to consistent geometric scales across diverse body shapes. 

To stabilize numerical ranges across channels, each time slice is normalized by the total heat trace ($\sum_k e^{-\lambda_k t}$) and log-transformed:

$$\widetilde{\text{HKS}}(x, t) = \log \frac{\text{HKS}(x, t)}{\sum_k e^{-\lambda_k t}}$$

The resulting **16-dimensional per-vertex descriptor** is strictly rotation-invariant and robust to articulation, serving as the primary input feature for the network.

![Heat Kernel Signature at increasing time scales](assets/hks.png)

### DiffusionNet

DiffusionNet (Sharp et al., 2022) turns heat diffusion into a learnable layer. HKS uses **fixed** times; DiffusionNet
diffuses learned features with **learned** times. The network is a stack of blocks, each of which:

1. **diffuses** every feature channel $u$ over the surface for its own learned time $t$, computed cheaply in the
   eigenbasis: $`u \mapsto \Phi\, \mathrm{diag}(e^{-\lambda t})\, \Phi^{\top} M\, u`$
   (small $t$ = local context, large $t$ = whole-body context);
2. computes **spatial gradient features**: inner products of surface gradients, which add a sense of direction
   (along a limb vs. across it) while staying rotation-invariant;
3. mixes the input, diffused and gradient features with a per-vertex **MLP** and a residual connection.

The network takes the per-vertex HKS as input and outputs 8 class scores per vertex. These are averaged over the three
vertices of each triangle to give one prediction per face, trained with cross-entropy against the face labels.

## Getting started

```bash
pip install torch scipy matplotlib robust_laplacian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Download the dataset, extract it to `human_seg/`, and run `main.ipynb`.

```
├── main.ipynb   # data loading, visualization, Laplace–Beltrami / HKS, model
├── human_seg/   # dataset (train/, test/, seg/, sseg/), not tracked by git
└── assets/      # figures used in this README
```

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG 2022. [code](https://github.com/nmwsharp/diffusion-net)
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP 2009.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes.* SGP 2020. [code](https://github.com/nmwsharp/robust-laplacians-py)
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH 2019. [code](https://github.com/ranahanocka/MeshCNN)
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH 2017.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* 2020. [code](https://github.com/facebookresearch/pytorch3d)
- Datasets: SCAPE (Anguelov et al., 2005), MIT animation (Vlasic et al., 2008), FAUST (Bogo et al., 2014), SHREC'07 (Giorgi et al., 2007).
