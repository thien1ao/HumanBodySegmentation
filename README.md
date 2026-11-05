# Human Body Mesh Segmentation with Spectral Features and DiffusionNet

Semantic segmentation of 3D human body meshes into eight anatomical parts: **head, hand, forearm, upper arm, torso, thigh, shin, foot**.

The network only sees **intrinsic** geometry: the Laplace–Beltrami eigenbasis and the Heat Kernel Signature (HKS). Because of this, its predictions do not depend on how the mesh is oriented or positioned in space.

<p align="center">
  <img src="assets/samples.png" alt="Ground-truth segmentations from the training set" height="600">
</p>

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

## Dataset

This project uses the preprocessed human segmentation benchmark from [MeshCNN](https://github.com/ranahanocka/MeshCNN)
([download](https://www.dropbox.com/scl/fi/r04ahap6vhuwzkjj9mpwk/human_seg.tar.gz?rlkey=dyjwx6ehxmarvtrzl6atvs9mz&e=1)). Extract it into the repository root:

```bash
tar -xzf human_seg.tar.gz 
```

| Split | Source             | Meshes |
|-------|--------------------|-------:|
| Train | MIT animation      | 169    |
| Train | FAUST              | 100    |
| Train | SCAPE              | 71     |
| Train | Adobe              | 41     |
| Test  | SHREC'07           | 18     |

## Methodology

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

<p align="center">
  <img src="https://img.shields.io/badge/Face%20Accuracy-88.1%25-2ea44f?style=for-the-badge" alt="Face accuracy 88.1%">
  <img src="https://img.shields.io/badge/Mean%20IoU-79.9%25-1f6feb?style=for-the-badge" alt="Mean IoU 79.9%">
  <img src="https://img.shields.io/badge/Test%20Set-SHREC%2707%20%C2%B7%2018%20meshes-6e7781?style=for-the-badge" alt="Test set SHREC'07, 18 meshes">
</p>

### 🎯 Qualitative results

Each row shows the ground truth, the prediction, and the misclassified faces (in red) for one unseen SHREC'07 mesh.

<p align="center">
  <img src="assets/predictions/shrec__1_prediction.png" alt="Prediction on SHREC'07 mesh 1" width="100%">
</p>

<details>
<summary><b>More examples</b></summary>
<br>
<p align="center">
  <img src="assets/predictions/shrec__10_prediction.png" alt="Prediction on SHREC'07 mesh 10" width="100%">
  <img src="assets/predictions/shrec__11_prediction.png" alt="Prediction on SHREC'07 mesh 11" width="100%">
</p>
</details>

### 📊 Per-class IoU

<div align="center">

| Part | IoU (%) | |
|:-----|:-------:|:--|
| ![head](https://img.shields.io/badge/head-CC4D4D?style=flat-square)           | 75.8 | `███████████████░░░░░` |
| ![hand](https://img.shields.io/badge/hand-4DB34D?style=flat-square)           | 84.7 | `█████████████████░░░` |
| ![forearm](https://img.shields.io/badge/forearm-4D73E6?style=flat-square)     | 84.0 | `█████████████████░░░` |
| ![upper arm](https://img.shields.io/badge/upper%20arm-F2BF33?style=flat-square) | 69.6 | `██████████████░░░░░░` |
| ![torso](https://img.shields.io/badge/torso-B359D9?style=flat-square)         | 82.3 | `████████████████░░░░` |
| ![thigh](https://img.shields.io/badge/thigh-33CCCC?style=flat-square)         | 75.7 | `███████████████░░░░░` |
| ![shin](https://img.shields.io/badge/shin-F28C26?style=flat-square)           | 82.5 | `████████████████░░░░` |
| ![foot](https://img.shields.io/badge/foot-999999?style=flat-square)           | 84.8 | `█████████████████░░░` |
| **Mean**                                                                      | **79.9** | `████████████████░░░░` |

</div>

The badge colors match the part colors in the figures. Hands, forearms and feet are segmented most reliably. The upper arm is the hardest part, because its boundary with the torso at the shoulder is the least well defined.

### 📈 Training curves

<p align="center">
  <img src="assets/history.png" alt="Training and test loss/accuracy" width="100%">
</p>

Test accuracy levels off at about 88% after roughly 15 epochs, while training accuracy keeps climbing to 96.6%. This points to mild overfitting.

### 🔍 Error analysis

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/confusion_matrix.png" alt="Confusion matrix" width="100%">
    </td>
    <td width="50%">
      <p>The confusion matrix is <b>row-normalized and area-weighted</b>. Almost all errors fall at boundaries between parts that touch:</p>
      <table>
        <tr><th>True → Predicted</th><th>Joint</th></tr>
        <tr><td>thigh → torso</td><td>🦵 hip</td></tr>
        <tr><td>upper arm → torso</td><td>💪 shoulder</td></tr>
        <tr><td>foot → shin</td><td>🦶 ankle</td></tr>
        <tr><td>hand → forearm</td><td>✋ wrist</td></tr>
        <tr><td>head → torso</td><td>🙂 neck</td></tr>
      </table>
      <p>These joints have no sharp geometric boundary, so where one part ends and the next begins is partly a labeling convention.</p>
    </td>
  </tr>
</table>

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG, 2022. [[code]](https://github.com/nmwsharp/diffusion-net)
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP, 2009.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes.* SGP, 2020. [[code]](https://github.com/nmwsharp/robust-laplacians-py)
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH, 2019. [[code]](https://github.com/ranahanocka/MeshCNN)
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH, 2017.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* arXiv, 2020. [[code]](https://github.com/facebookresearch/pytorch3d)
- Datasets: SCAPE (Anguelov et al., 2005), MIT animation (Vlasic et al., 2008), FAUST (Bogo et al., 2014), SHREC'07 (Giorgi et al., 2007).
