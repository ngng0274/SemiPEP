# SemiPEP: Semi-supervised Learning Framework for Paratope and Epitope Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository is the official implementation of the research on enhancing antibody-antigen binding site prediction using Semi-supervised Learning (SSL). **SemiPEP** effectively leverages large-scale unlabeled data to overcome the scarcity of experimentally labeled complexes.

## Overview
Predicting the binding sites of antibodies (paratopes) and antigens (epitopes) is a fundamental task in therapeutic discovery. **SemiPEP** addresses the data bottleneck by utilizing a dual-model interaction framework that progressively learns from unlabeled data through high-confidence pseudo-labeling.

### Key Innovations:
* **Combined Confidence Estimation**: Integrates global graph-level features with **k-hop subgraph-based local confidence** to rigorously evaluate the reliability of pseudo-labels.
* **Adaptive Percentile-based Thresholding**: Instead of using a static threshold, it employs a dynamic percentile scheduling strategy ($p_{min}$ to $p_{max}$) to progressively incorporate high-quality pseudo-labels.
* **Joint Prediction**: Utilizes Cross-Attention mechanisms within a GAT-based architecture to capture the mutual structural interactions between antibodies and antigens.

## Architecture
The framework consists of two interacting components:
1.  **Target Model ($f$)**: A GAT-based encoder that extracts structural features and models interactions via Cross-Attention to predict binding probabilities.
2.  **Instructor Model ($g$)**: An MLP-based model that estimates prediction confidence, utilizing a learnable parameter $\alpha$ to balance global and local structural information.

## Dataset: AsEP (Modified)
We utilize the **AsEP dataset**, the most comprehensive benchmark for antibody-specific epitope prediction.
* **Source**: The original dataset is provided by [biochunan/AsEP-dataset](https://github.com/biochunan/AsEP-dataset).
* **Modified Pre-processing**: The original dataset loader and pre-processing logic were modified to support the **IgFold** (Antibody) and **ESM-2** (Antigen) embedding integration required for this framework.
* **Multi-modal Features**: Combines pre-calculated protein language model (PLM) embeddings with one-hot encoded amino acid features.


## Getting Started

### 1. Installation
Install the dependencies using the provided `requirements.txt`. This project is optimized for **PyTorch 2.8.0 with CUDA 12.9**.

```
pip install -r requirements.txt
```

### 2. Dataset Setup
Organize the modified AsEP dataset in the following structure:
```
data/
└── AsEP/
    ├── raw/
    └── processed/
```

### 3. Running SemiPEP
The `SemiPEP.py` script handles both pre-training and the main semi-supervised training loop (Algorithm 1).

```
python SemiPEP.py --gpu 0 --num_epochs 100 --batch_size 32 --data_path ./data
```

### 📁 Repository Structure
```
├── asep/                   # Modified dataset loader and pre-processing scripts for AsEP.
├── model.py                # Definitions for SemiPEP_Target and SemiPEP_InstructorMLP.
├── SemiPEP.py              # Main execution script implementing the adaptive thresholding algorithm.
├── utils.py                # Evaluation utilities for AUROC, AUPRC, and MCC metrics.
├── requirements.txt
└── README.md
```

## Publication
- Master's Thesis: [Semi-supervised Learning Framework for Paratope and Epitope Prediction (2025)](https://postech.dcollection.net/public_resource/pdf/200000895760_20260308162818.pdf)
- Conference Paper: [항체 항원 결합부위 예측을 위한 준지도 학습 기법](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12318274)