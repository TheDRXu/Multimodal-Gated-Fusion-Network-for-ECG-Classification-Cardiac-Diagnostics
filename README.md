# Multimodal-Gated-Fusion-Network-for-ECG-Classification-Cardiac-Diagnostics

A deep learning pipeline for multi-label cardiac diagnosis using the **PTB-XL ECG dataset**. This project combines 12-lead ECG signals with patient tabular metadata (age, sex, height, weight) in a dual-branch neural network to classify five diagnostic superclasses.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Diagnostic Classes](#diagnostic-classes)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)

---

## Overview

This project tackles **multi-label ECG classification** — where a single ECG recording can belong to multiple diagnostic categories simultaneously. The approach fuses two input modalities:

- **ECG signals**: 12-lead waveforms at 100 Hz (shape: `12 × 1000`)
- **Tabular metadata**: age, sex, height, weight (normalized)

Class imbalance is addressed using **MLSMOTE** (Multi-Label SMOTE), a synthetic oversampling technique adapted for multi-label settings.

---

## Dataset

**PTB-XL** — a large publicly available 12-lead ECG dataset.

| Property | Value |
|---|---|
| Source | [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) |
| Sampling rate | 100 Hz (low-res) |
| Signal shape | `(N, 12, 1000)` |
| Labels | Multi-label, 5 diagnostic superclasses |

Download the dataset and set the path in the notebook:
```python
path = "/path/to/ptbxl/"
```

The loader reads `ptbxl_database.csv` and `scp_statements.csv`, extracts diagnostic superclasses, and returns ECG signals, tabular features, and binary label vectors.

---

## Model Architecture

### `MultiInputECGNet`

A two-branch architecture with a gated fusion layer:

```
ECG Input (12 × 1000)
       │
   [YBranch]
   Stem Conv → ResidualSEBlock × 3 → Multi-scale Conv → GlobalAvgPool
       │                                                        │
       └──────────────────────── Concatenate (448-d) ──────────┘
                                        │
Tabular Input (4-d)              [Fusion Gate]
       │                          (Sigmoid-weighted)
   [XBranch]                            │
   Linear → BN → ReLU × 2        [Classifier]
                               Linear(448→256→128→5)
                                        │
                                  5-class Output
```

**Key components:**

- **ResidualSEBlock** — Residual block with Squeeze-and-Excitation (channel attention)
- **YBranch (ECG)** — CNN with multi-scale convolutional heads (`kernel 3, 5, 7`) producing a 384-d feature vector
- **XBranch (Tabular)** — Two-layer MLP producing a 64-d feature vector
- **Fusion Gate** — Learned sigmoid gate that reweights the concatenated feature vector before classification

---

## Diagnostic Classes

| Index | Code | Description |
|---|---|---|
| 0 | `NORM` | Normal ECG |
| 1 | `MI` | Myocardial Infarction |
| 2 | `STTC` | ST/T-wave Change |
| 3 | `CD` | Conduction Disturbance |
| 4 | `HYP` | Hypertrophy |

---

## Installation

```bash
git clone https://github.com/your-username/ecg-classification.git
cd ecg-classification

pip install torch torchvision wfdb numpy pandas scikit-learn matplotlib seaborn
```

**Requirements:**

- Python 3.8+
- PyTorch 1.12+
- `wfdb` (for reading PhysioNet records)
- scikit-learn, numpy, pandas, matplotlib, seaborn

---

## Usage

1. Download the PTB-XL dataset from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
2. Update the dataset path in the notebook
3. Run cells sequentially in `ecg.ipynb`

```python
path = "/path/to/ptbxl/"
X_tab, X, Y = load_ptbxl(path)
```

---

## Training

The model is trained with:

| Hyperparameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` (multi-label) |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Epochs | 40 |
| Batch size | 64 |
| Train / Val / Test split | 80 / 10 / 10 % |

The best model checkpoint (by validation ROC-AUC) is saved as `best_multimodal_ecgnet.pth`.

---

## Evaluation

**ROC-AUC: 93.04%**

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| NORM | 0.8881 | 0.8185 | 0.8519 | 931 |
| MI | 0.6872 | 0.8305 | 0.7521 | 537 |
| STTC | 0.7917 | 0.6884 | 0.7364 | 552 |
| CD | 0.8012 | 0.7842 | 0.7926 | 519 |
| HYP | 0.7709 | 0.5149 | 0.6174 | 268 |

---

## Project Structure

```
ecg-classification/
├── ecg.ipynb               # Main notebook
├── best_multimodal_ecgnet.pth  # Saved model checkpoint (generated after training)
└── README.md
```

---

## Acknowledgements

- Dataset: [PTB-XL by Wagner et al., 2020](https://www.nature.com/articles/s41597-020-0495-6)
- PhysioNet: [physionet.org](https://physionet.org)
