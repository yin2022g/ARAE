# ARAE: Adaptive ReAdjust Edge-weight Framework

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**ARAE** is a Graph Neural Network (GNN) framework designed for **spatial domain recognition** in spatial transcriptomics data. It enhances domain identification through an adaptive edge-weight adjustment mechanism and a specialized deconvolution module for low-resolution data.

## 📖 Overview

Spatial transcriptomics enables the measurement of gene expression with spatial context. ARAE improves the accuracy of spatial domain identification by:

1.  **Adaptive Edge Adjustment**: Dynamically redistributing edge weights based on embedding similarity to distinguish neighbor information.
2.  **Deconvolution Module**: specifically improving resolution and accuracy for low-resolution datasets.
3.  **Graph Convolutional Network (GCN)**: Integrating neighborhood information for robust spatial embedding.

<!-- 如果你有模型架构图，请把图片放在项目中并替换下面的路径 -->
<!-- ![Model Architecture](images/framework.png) -->

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yin2022g/ARAE.git
cd ARAE
```

### 2. Install dependencies
It is recommended to use a virtual environment (Conda or venv).
```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

The model has been validated on multiple public datasets. Please download the data and organize it into the `data/` folder.

| Dataset Name | Platform | Link |
| :--- | :--- | :--- |
| **Human DLPFCs** | spatialLIBD | [Download](http://spatial.libd.org/spatialLIBD) |
| **Mouse Anterior Brain** | 10x Visium | [Download](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Mouse_Brain_Sagittal_Anterior) |
| **Human Breast Cancer** | 10x Visium | [Download](https://www.10xgenomics.com/cn/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0) |
| **Mouse Olfactory** | Slide-seq | [Download](https://portals.broadinstitute.org/single_cell/study/slide-seq-study) |
| **Mouse Posterior Brain** | 10x Visium | [Download](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.0.0/V1_Mouse_Brain_Sagittal_Posterior) |

---

## 🚀 Usage

### 1. Data Preprocessing
Process the raw data into the required format:
```bash
python preprocess.py
```

### 2. Training and Evaluation
Run the main script to train the model and generate plots:
```bash
python test_find_best_plot.py
```

*Note: You may need to adjust parameters inside `config/` or pass arguments depending on your specific setup.*

---

## 📂 Project Structure

```text
ARAE/
├── config/                  # Configuration parameters
├── data/                    # Dataset storage
├── layers.py                # Neural network layers (GCN, etc.)
├── models.py                # Main ARAE model definition
├── preprocess.py            # Data preprocessing scripts
├── test_find_best_plot.py   # Training and testing entry point
├── utils.py                 # Helper functions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🤝 Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{ARAE2025,
  title={ARAE: Adaptive ReAdjust Edge-weight framework for spatial domain recognition},
  author={Yin, G. and Authors},
  journal={Journal Name},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.