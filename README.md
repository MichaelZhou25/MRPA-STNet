# MRPA-STNet

Core PyTorch implementation for the manuscript:

**Multi-Resolution Phase-Aware Spatio-temporal Network for EEG-Based Epileptic Seizure Classification**

## Overview

This repository provides the **core neural-network implementation** of the proposed seizure-classification framework. The released code contains the model architecture described in the manuscript, including:

1. multi-modal frequency-feature fusion;
2. a frequency branch for local spectral representation learning;
3. a residual temporal-convolutional branch;
4. the Channel-Attention Gated Recurrent Unit (CA-GRU);
5. temporal attention pooling; and
6. the final binary classification head.

The repository is intentionally limited to the core model implementation and its documentation. Dataset redistribution, preprocessing pipelines, training scripts, cross-validation scripts, and evaluation scripts are not included.

## Repository Structure

```text
MRPA-STNet/
├── model.py
└── README.md
```

## Requirements

The released `model.py` has only one external Python dependency:

- Python 3.9 or later
- PyTorch

Install PyTorch using the installation command appropriate for your operating system and CUDA environment. For a CPU-only environment, a typical installation is:

```bash
pip install torch
```

> Note: the original experimental environment was not distributed as a locked environment file, so an exact PyTorch version is not claimed here.

## Model Input

`model.py` implements the neural network only. It **does not directly read raw EEG/EDF files** and does not perform the STFT preprocessing described in the paper.

The model expects a pre-extracted tensor with shape:

```text
(B, T, C, Freq + Time)
```

where:

- `B`: batch size;
- `T`: number of aligned temporal frames;
- `C`: number of EEG channels;
- `Freq`: frequency-feature dimension;
- `Time`: temporal-feature dimension.

For the default configuration used by the released model:

```text
Freq = 192
Time = 256
Total feature dimension = 448
```

The first 192 dimensions correspond to three frequency-domain modalities, each represented by 64 bins:

```text
[long-window magnitude | long-window phase | short-window magnitude]
```

The remaining 256 dimensions contain the temporal segment aligned with the corresponding time-frequency frame.

Thus, the default input shape is:

```text
(B, T, C, 448)
```

## Architecture

### 1. Multi-modal frequency fusion

The three 64-dimensional frequency modalities are reshaped into modality tokens and fused using multi-head self-attention. The fused representation is normalized and passed to the frequency branch.

### 2. Frequency branch

A lightweight 1D convolutional block learns local spectral patterns from the fused 64-dimensional representation and outputs a 64-dimensional frequency feature for each channel and temporal frame.

### 3. Temporal branch

The temporal representation is processed by two residual temporal-convolutional blocks with dilation factors 1 and 2. Global averaging along the temporal-feature axis produces a 64-dimensional temporal feature.

### 4. CA-GRU

The 64-dimensional frequency feature and 64-dimensional temporal feature are concatenated to form a 128-dimensional representation. CA-GRU then models recurrent dynamics while applying channel self-attention within the reset, update, and candidate-state computations.

### 5. Temporal attention pooling

The recurrent output is first averaged across EEG channels. A learnable temporal-attention module then assigns weights to temporal frames and produces a 64-dimensional sample-level representation.

### 6. Classification head

The pooled representation is passed through a two-layer fully connected classifier to produce logits for seizure/non-seizure classification.

## Basic Usage

```python
import torch
from model import MainModel

config = {
    "Freq": 192,
    "Time": 256,
    "DROPOUT": 0.3,
    "ATTN_DROPOUT": 0.1,
    "TIME_ATTN_DROPOUT": 0.1,
    "TIME_POOL": "attn",
}

model = MainModel(
    Freq=192,
    Time=256,
    num_classes=2,
    config=config,
)

# Example only: pre-extracted input tensor
# B=2, T=3 aligned frames, C=18 EEG channels, F=192+256=448
x = torch.randn(2, 3, 18, 448)

model.eval()
with torch.no_grad():
    logits = model(x)

print(logits.shape)  # torch.Size([2, 2])
```

## Main Configuration Options

| Key | Default | Description |
|---|---:|---|
| `Freq` | `192` | Total frequency-domain feature dimension |
| `Time` | `256` | Temporal-feature dimension |
| `DROPOUT` | `0.5` if unspecified | Dropout in the classifier |
| `ATTN_DROPOUT` | `0.1` | Dropout applied to CA-GRU channel-attention weights |
| `TIME_ATTN_DROPOUT` | `0.1` | Dropout in temporal attention |
| `TIME_ATTN_HIDDEN` | `32` | Hidden dimension of the temporal-attention network |
| `TIME_POOL` | `attn` | Temporal pooling mode: `attn`, `mean`, or `last` |

The implementation also retains two ablation switches used by the authors:

- `ABLATION_STANDARD_GRU=True`: replace CA-GRU with a standard GRU;
- `ABLATION_NO_MODAL_ATTN=True`: disable multi-modal attention fusion.

## Data Availability

The CHB-MIT and Siena scalp EEG datasets are **not redistributed** with this repository. Users should obtain the datasets from their respective official sources and comply with the corresponding data-use requirements.

## Scope of the Release

This public repository provides the **core model implementation** needed to inspect and reuse the proposed architecture. It should not be interpreted as a complete reproduction package for all experiments reported in the manuscript because preprocessing, dataset construction, training, cross-validation, and evaluation pipelines are outside the scope of this release.

## Citation

If you use this implementation in academic work, please cite the associated paper after publication.

```text
Citation information will be added after publication.
```

## Contact

For questions regarding the model architecture, please contact the corresponding authors listed in the manuscript.
