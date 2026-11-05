# 3D Human Mesh Part Segmentation with Spectral Features and DiffusionNet

Semantic segmentation of human body meshes into 8 parts (head, hand, forearm, upper arm, torso, thigh, shin, foot).
The network only sees **intrinsic** geometry (Laplace–Beltrami eigenbasis and Heat Kernel Signature), so its
predictions do not depend on how the mesh is oriented.

**Test results SHREC'07: 88.1% face accuracy, 79.9% mIoU**

<img src="assets/samples.png" alt="samples" height="600">

## Repository Structure
The preprocessed human segmentation benchmark from [MeshCNN](https://github.com/ranahanocka/MeshCNN)
([download](https://www.dropbox.com/scl/fi/r04ahap6vhuwzkjj9mpwk/human_seg.tar.gz?rlkey=dyjwx6ehxmarvtrzl6atvs9mz&e=1)),
extracted to `human_seg/`.

```
HumanBodySegmentation/
├── python_scripts/
│   ├── data.py        # Mesh loading, labels and alignment
│   ├── features.py    # Spectral decomposition and HKS
│   ├── model.py       # MiniDiffusionNet architecture
│   ├── train.py       # Training loop
│   ├── evaluate.py    # Evaluation and confusion matrix
│   └── plotting.py    # Visualization utilities
│
├── notebooks/
│   └── main.ipynb     # Data exploration and experiments
│
├── assets/
├── human_seg/         # Extracted dataset
└── model.pt           # Trained model
```

## Methodology

![pipeline](assets/pipeline.png)

### MiniDiffusionNet

```python
import torch
import torch.nn as nn


class SpectralDiffusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.t = nn.Parameter(torch.zeros(channels))

    def forward(self, x, evals, evecs, mass):
        coefs = evecs.T @ (mass[:, None] * x)
        decay = torch.exp(-evals[:, None] * self.t.abs()[None, :])
        return evecs @ (decay * coefs)


class DiffusionBlock(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.diffusion = SpectralDiffusion(channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )

    def forward(self, x, evals, evecs, mass):
        x_diffused = self.diffusion(x, evals, evecs, mass)
        return x + self.mlp(torch.cat([x, x_diffused], dim=-1))


class MiniDiffusionNet(nn.Module):
    def __init__(self, in_ch, num_classes,
                 width=128, n_blocks=4, dropout=0.3):
        super().__init__()
        self.inp = nn.Linear(in_ch, width)
        self.blocks = nn.ModuleList(
            DiffusionBlock(width, dropout)
            for _ in range(n_blocks)
        )
        self.out = nn.Linear(width, num_classes)

    def forward(self, x, evals, evecs, mass, faces):
        h = self.inp(x)

        for blk in self.blocks:
            h = blk(h, evals, evecs, mass)

        vertex_logits = self.out(h)
        return vertex_logits[faces].mean(1)
```

## Results

### Training performance

![Training curves](assets/history.png)

*Training and test loss/accuracy. Test accuracy plateaus around 88% after ~15 epochs while training accuracy keeps
rising to 96.6%, so the model overfits slightly.*

### Confusion matrix

<img src="assets/confusion_matrix.png" alt="Confusion matrix" width="400">

*Row-normalized, area-weighted confusion matrix. Almost all errors are between parts that touch:
thigh → torso (hip), upper arm → torso (shoulder), foot → shin (ankle), hand → forearm (wrist), head → torso (neck).*

### Predictions

<img src="assets/predictions/shrec__10_prediction.png" alt="preditiction_10">

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG 2022. [code](https://github.com/nmwsharp/diffusion-net)
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP 2009.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes.* SGP 2020. [code](https://github.com/nmwsharp/robust-laplacians-py)
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH 2019. [code](https://github.com/ranahanocka/MeshCNN)
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH 2017.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* 2020. [code](https://github.com/facebookresearch/pytorch3d)
- Datasets: SCAPE (Anguelov et al., 2005), MIT animation (Vlasic et al., 2008), FAUST (Bogo et al., 2014), SHREC'07 (Giorgi et al., 2007).


