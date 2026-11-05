# 3D Human Mesh Part Segmentation with Spectral Features and DiffusionNet

Semantic segmentation of human body meshes into 8 parts (head, hand, forearm, upper arm, torso, thigh, shin, foot).
The network only sees **intrinsic** geometry (Laplace–Beltrami eigenbasis and Heat Kernel Signature), so its
predictions do not depend on how the mesh is oriented.

**Test results SHREC'07: 88.1% face accuracy, 79.9% mIoU**

![Sample meshes with ground-truth body-part labels](assets/samples.png)

## Dataset

The preprocessed human segmentation benchmark from [MeshCNN](https://github.com/ranahanocka/MeshCNN)
([download](https://www.dropbox.com/scl/fi/r04ahap6vhuwzkjj9mpwk/human_seg.tar.gz?rlkey=dyjwx6ehxmarvtrzl6atvs9mz&e=1)),
extracted to `human_seg/`.

| Split | Source | Meshes |
|-------|--------|-------:|
| train | MIT animation (Vlasic et al., 2008) | 169 |
| train | FAUST (Bogo et al., 2014) | 100 |
| train | SCAPE (Anguelov et al., 2005) | 71 |
| train | Adobe Fuse | 41 |
| test  | SHREC'07 (Giorgi et al., 2007) | 18 |

## Methodology

![pipeline](assets/pipeline.png)

### MiniDiffusionNet

[DiffusionNet](https://github.com/nmwsharp/diffusion-net) (Sharp et al., 2022) turns heat diffusion into a learnable
layer: HKS uses **fixed** times, DiffusionNet diffuses learned features with **learned** times. This project uses a
compact version of it:

Each **DiffusionBlock**:

1. **diffuses** every channel $u$ over the surface for its own learned time $t_c$, computed in the eigenbasis:
   $`u \mapsto \Phi\, \mathrm{diag}(e^{-\lambda t_c})\, \Phi^{\top} M\, u`$
   (small $t_c$ = local context, large $t_c$ = whole-body context);
2. concatenates the input and diffused features and mixes them with a per-vertex **MLP**
   (256 → 128 → 128, ReLU, dropout 0.3);
3. adds the result back to the input (**residual** connection).

Unlike the full DiffusionNet, there are no spatial-gradient features; the model only has isotropic diffusion.

**Training:** cross-entropy on face labels, AdamW (lr 1e-3, weight decay 1e-4), cosine schedule over 100 epochs,
one mesh per step, Gaussian noise (σ = 0.05) on the standardized HKS as augmentation.

## Results

Evaluated on the 18 SHREC'07 test meshes after 100 epochs.

![Training curves](assets/history.png)

*Training and test loss/accuracy. Test accuracy plateaus around 88% after ~15 epochs while training accuracy keeps
rising to 96.6%, so the model overfits slightly.*

![Confusion matrix](assets/confusion_matrix.png)

*Row-normalized, area-weighted confusion matrix. Almost all errors are between parts that touch:
thigh → torso (hip), upper arm → torso (shoulder), foot → shin (ankle), hand → forearm (wrist), head → torso (neck).*

## Getting started

```bash
pip install torch scipy matplotlib robust_laplacian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Download the dataset and extract it to `human_seg/`, then:

```bash
python python_scripts/train.py     # preprocess, train for 100 epochs, save model.pt (~2 min on a GPU)
```

`model.pt` stores the weights together with the HKS times and normalization statistics, which `predict.py` and
`annotate.py` need to process new meshes.

```
├── python_scripts/
│   ├── data.py        # paths, mesh loading, edge → face labels, alignment
│   ├── features.py    # Laplace–Beltrami eigenbasis, HKS, normalization
│   ├── model.py       # MiniDiffusionNet
│   ├── train.py       # training loop; run to train and save model.pt
│   ├── evaluate.py    # predictions, confusion matrix, IoU
│   ├── plotting.py    # all figures
│   ├── predict.py     # segment any .obj with the trained model
│   └── annotate.py    # write .eseg / .seseg annotation files from predictions
├── notebooks/
│   └── main.ipynb     # exploration notebook: data, spectra, HKS, training, analysis
├── annotations/       # generated annotation files for new meshes
├── assets/            # figures used in this README
├── human_seg/         # dataset (train/, test/, seg/, sseg/), not tracked by git
└── model.pt           # trained model
```

## References

- N. Sharp, S. Attaiki, K. Crane, M. Ovsjanikov. *DiffusionNet: Discretization Agnostic Learning on Surfaces.* ACM TOG 2022. [code](https://github.com/nmwsharp/diffusion-net)
- J. Sun, M. Ovsjanikov, L. Guibas. *A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion.* SGP 2009.
- N. Sharp, K. Crane. *A Laplacian for Nonmanifold Triangle Meshes.* SGP 2020. [code](https://github.com/nmwsharp/robust-laplacians-py)
- R. Hanocka et al. *MeshCNN: A Network with an Edge.* SIGGRAPH 2019. [code](https://github.com/ranahanocka/MeshCNN)
- H. Maron et al. *Convolutional Neural Networks on Surfaces via Seamless Toric Covers.* SIGGRAPH 2017.
- N. Ravi et al. *Accelerating 3D Deep Learning with PyTorch3D.* 2020. [code](https://github.com/facebookresearch/pytorch3d)
- Datasets: SCAPE (Anguelov et al., 2005), MIT animation (Vlasic et al., 2008), FAUST (Bogo et al., 2014), SHREC'07 (Giorgi et al., 2007).

