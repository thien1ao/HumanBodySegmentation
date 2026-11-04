# 3D Human Mesh Part Segmentation using DiffusionNet

## Overview

A deep learning pipeline for semantic 3D human mesh segmentation (8 classes) that leverages intrinsic geometric features (Laplace–Beltrami and HKS) to achieve spatial invariance, ensuring robust generalization across diverse poses and datasets.

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

On a flat grid the Laplacian $\Delta f$ measures how much a value differs from the average of its neighbors.
The **Laplace–Beltrami operator** is the same idea on a curved surface. On a triangle mesh it is discretized as the
**cotangent Laplacian**:

$$ (L f)_i = \sum_{j \in N(i)} w_{ij}\,(f_i - f_j), \qquad w_{ij} = \tfrac12(\cot\alpha_{ij} + \cot\beta_{ij}) $$

where $\alpha_{ij}, \beta_{ij}$ are the two angles opposite the edge $(i,j)$. Each vertex also gets a **mass**
$M_i$ = one third of the area of its adjacent triangles (the "lumped mass matrix").

Only angles and areas appear, and those don't change under rotation, so $L$ is rotation-invariant;
it also changes very little when the body bends, because bending barely stretches the surface.

**Eigenfunctions.** Solving the generalized eigenproblem

$$ L\,\phi_k = \lambda_k\, M\, \phi_k, \qquad 0 = \lambda_0 \le \lambda_1 \le \dots $$

gives functions $\phi_k$ on the surface that are the mesh's "vibration modes" (a Fourier basis for the surface):
small $\lambda$ = smooth, large-scale patterns; large $\lambda$ = fine detail.
With only ~750 vertices we can just do a dense eigendecomposition. We keep the first `K_EIG = 128` modes.

Implementation trick: with $\psi = M^{1/2}\phi$ the problem becomes the ordinary symmetric eigenproblem
$M^{-1/2} L M^{-1/2}\,\psi = \lambda\,\psi$, which `torch.linalg.eigh` solves directly.

$$
L\\phi_k = \lambda_k\ M\ \phi_k
$$

![Laplace–Beltrami eigenfunctions on a body mesh](assets/laplace.png)

**Heat Kernel Signature**:

$$
\mathrm{HKS}(x,t) = \sum_k e^{-\lambda_k t}\ \phi_k(x)^2
$$


![Heat Kernel Signature at increasing time scales](assets/hks.png)


**DiffusionNet**:

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG 2022. [code](https://github.com/nmwsharp/diffusion-net)
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP 2009.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes.* SGP 2020. [code](https://github.com/nmwsharp/robust-laplacians-py)
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH 2019. [code](https://github.com/ranahanocka/MeshCNN)
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH 2017.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* 2020. [code](https://github.com/facebookresearch/pytorch3d)
- Datasets: SCAPE (Anguelov et al., 2005), MIT animation (Vlasic et al., 2008), FAUST (Bogo et al., 2014), SHREC'07 (Giorgi et al., 2007).
