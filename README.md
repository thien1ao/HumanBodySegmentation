# 3D Human Body Mesh Part Segmentation with Spectral Features and DiffusionNet

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

## Pipeline

![pipeline](assets/pipeline.png)


## Results

### Predictions

<img src="assets/predictions/shrec__1_prediction.png" alt="preditiction_1">
<img src="assets/predictions/shrec__10_prediction.png" alt="preditiction_10">
<img src="assets/predictions/shrec__11_prediction.png" alt="preditiction_11">

### Training performance

![Training curves](assets/history.png)

*Training and test loss/accuracy. Test accuracy plateaus around 88% after ~15 epochs while training accuracy keeps
rising to 96.6%, so the model overfits slightly.*


