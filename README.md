# 3D Human Mesh Part Segmentation using DiffusionNet

A deep learning pipeline for semantic 3D human mesh segmentation (8 classes) that leverages intrinsic geometric features (Laplace–Beltrami and HKS) to achieve spatial invariance, ensuring robust generalization across diverse poses and datasets.

![Sample meshes with ground-truth body-part labels](assets/samples.png)

## Pipeline

```
mesh (.obj) ──► normalize ──► Laplace–Beltrami ──► Heat Kernel ──► DiffusionNet ──► label
                (center,       eigenbasis          Signature                       per face
                 unit area)    (λ, Φ)              (16 scales)
```

The meshes come from five datasets, and each one uses its own pose, body shape and coordinate frame. Instead of raw `xyz` coordinates, every step works with **intrinsic** geometry: quantities measured along the surface. These quantities don't change when a body is rotated, moved or rescaled, and they barely change when it moves into another pose.

## Dataset

The preprocessed human segmentation benchmark from [MeshCNN](https://github.com/ranahanocka/MeshCNN) ([download](https://www.dropbox.com/scl/fi/r04ahap6vhuwzkjj9mpwk/human_seg.tar.gz?rlkey=dyjwx6ehxmarvtrzl6atvs9mz&e=1)), extracted to `human_seg/`.

| Split | Source | Meshes |
|-------|--------|-------:|
| train | MIT animation (Vlasic et al., 2008) | 169 |
| train | FAUST (Bogo et al., 2014) | 100 |
| train | SCAPE (Anguelov et al., 2005) | 71 |
| train | Adobe Fuse | 41 |
| test  | SHREC'07 (Giorgi et al., 2007) | 18 |

Every mesh has **752 vertices and 1500 faces**. The labels are stored **per edge** (`seg/*.eseg`). We convert them to **per face** by majority vote over each triangle's three edges.

| head | hand | forearm | upper arm | torso | thigh | shin | foot |
|:----:|:----:|:-------:|:---------:|:-----:|:-----:|:----:|:----:|
| 7.1 % | 4.9 % | 7.3 % | 8.7 % | 31.3 % | 16.2 % | 15.5 % | 9.1 % |

The classes are imbalanced: the torso covers almost a third of the surface, the hands less than 5 %. The labels don't separate left from right.

## Method

**Normalization.** Each mesh is centered at its area-weighted centroid and scaled to unit surface area, `V ← V / √A`. After this, the spectra of different meshes can be compared directly.

**Laplace–Beltrami operator.** We build the cotangent Laplacian `L` and the mass matrix `M` with `robust_laplacian`, then solve for the first 128 eigenpairs:

$$
L\,\phi_k = \lambda_k\, M\,\phi_k
$$

The eigenfunctions `φ_k` are the surface version of Fourier modes: low `k` gives smooth variation over the whole body, high `k` gives fine detail.

**Heat Kernel Signature.** For each vertex, HKS measures how much heat stays at that vertex after time `t`:

$$
\mathrm{HKS}(x,t) = \sum_k e^{-\lambda_k t}\,\phi_k(x)^2
$$

Small `t` captures local shape, such as fingertips, toes and the head. Large `t` captures the vertex's position within the whole body. We use 16 log-spaced times in `[0.01, 1]`, normalize each by the heat trace and take the log. The result is a 16-dimensional descriptor per vertex.

**DiffusionNet.** The network (Sharp et al., 2022) stacks blocks that each:
1. **diffuse** every feature channel over the surface for a **learned** time `t`, computed in the eigenbasis as `Φ diag(e^{-λt}) Φᵀ M u`;
2. compute **spatial gradient features**, which add a sense of direction (along a limb versus across it);
3. mix the results with a per-vertex **MLP** and a residual connection.

It takes HKS as input and outputs class scores for each vertex. These are averaged over each triangle's vertices to get one prediction per face, which is trained with cross-entropy.

## Getting started

```bash
pip install torch scipy matplotlib robust_laplacian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Download and extract the dataset into `human_seg/`, then run `main.ipynb`.

```
├── main.ipynb      # data loading, visualization, LBO / HKS, model
├── human_seg/      # dataset: train/, test/, seg/, sseg/
└── assets/         # figures
```

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG 2022. [code](https://github.com/nmwsharp/diffusion-net)
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP 2009.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes.* SGP 2020. [code](https://github.com/nmwsharp/robust-laplacians-py)
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH 2019. [code](https://github.com/ranahanocka/MeshCNN)
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH 2017.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* 2020. [code](https://github.com/facebookresearch/pytorch3d)
- Datasets: SCAPE (Anguelov et al., 2005), MIT animation (Vlasic et al., 2008), FAUST (Bogo et al., 2014), SHREC'07 (Giorgi et al., 2007).
