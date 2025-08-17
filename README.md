# 🌍 Deconfounding Multi-Cause Latent Confounders: A Factor-Model Approach to Climate Model Bias Correction

[![IJCAI 2025](https://img.shields.io/badge/IJCAI-2025-blue.svg)](https://ijcai-25.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🏆 **Accepted at IJCAI 2025**

This repository contains the official implementation of our IJCAI 2025 paper: *"Deconfounding Multi-Cause Latent Confounders: A Factor-Model Approach to Climate Model Bias Correction."*

## 📖 Overview

Our innovative approach addresses climate model bias correction through a novel two-stage methodology that leverages causal inference principles. Unlike traditional methods, our approach doesn't assume all confounding variables are observed, making it more robust for real-world climate modeling scenarios.

### 🎯 Key Features

- **Novel Deconfounding Approach**: First application of causal inference deconfounding to climate bias correction
- **Factor Model Architecture**: Advanced latent confounder modeling for unobserved variables
- **State-of-the-art Performance**: Integration with iTransformer for superior correction results
- **Real-world Application**: Validated on South Australia climate data

## 🗺️ Study Area

Our research focuses on **South Australia**, chosen for its diverse climate patterns and data availability.

<div align="center">
  <img src="figures/Study_area_NCEP_2.png" alt="Study Area - South Australia" width="60%">
  <p><em>Study Area: South Australia with NCEP grid points</em></p>
</div>

## 🔄 Methodology

Our method is divided into two complementary stages:

<div align="center">
  <img src="figures/Process_final_2.png" alt="Two-Stage Process Overview" width="70%">
  <p><em>Overview of our two-stage methodology: Deconfounding + Correction</em></p>
</div>

### Stage 1: Deconfounding 🧠
Identifies and models latent confounders using causal inference principles

### Stage 2: Correction 🎯
Applies bias correction using the learned latent features with iTransformer

## 📊 Theoretical Foundation

### Causal Graph Structure

Our approach is based on the following summary causal graph:

<div align="center">
  <img src="figures/Summary%20causal%20graph_final.png" alt="Summary Causal Graph" width="60%">
  <p><em>Summary causal graph showing the relationships between variables</em></p>
</div>

### Double Source Architecture

<div align="center">
  <img src="figures/double_source.png" alt="Double Source Architecture" width="40%">
  <p><em>Double source data integration approach</em></p>
</div>

### Factor Model Structure

<div align="center">
  <img src="figures/factor%20model.png" alt="Deconfounding BC Factor Model" width="100%">
  <p><em>Detailed structure of our Deconfounding BC factor model</em></p>
</div>

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### 📁 Data Preparation

#### Option 1: Simulation Data 🎲

Generate synthetic datasets for testing and validation:

```bash
# Generate simulated datasets
python main_run_simulation.py

# Process results and convert to CSV
python result_process.py
```

#### Option 2: Real-World Data 🌐

Download climate data from official sources:

**Data Sources:**
- 🌡️ **IPSL CMIP6**: [AIMS2 Portal](https://aims2.llnl.gov/search/cmip6)
- 🌊 **NCEP-NCAR Reanalysis**: [NOAA Portal](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)

**Download Instructions:**
1. Select surface level and 2-meter variables
2. Download monthly data in NetCDF (`.nc`) format
3. Convert to CSV format
4. Extract South Australia region data
5. Clean datasets by removing columns with excessive missing data

> **📝 Note:** For IPSL data, download all experimental settings from `r1p1i1f1` to `r33p1i1f1`

## 🔬 Stage 1: Deconfounding

The Deconfounding stage introduces causal inference insights to climate bias correction, handling unobserved confounders effectively.

### Running Deconfounding

```bash
cd Deconfounding/
python main_deconfounding.py
```

## 🎯 Stage 2: Correction

The Correction stage leverages the learned latent confounders as additional features for precipitation correction using the state-of-the-art iTransformer model.

### Running Correction

```bash
# Place processed data in dataset/weather folder
bash ./scripts/multivariate_forecasting/Weather/bc_iTransformer.sh
```

> **🔗 Based on:** [iTransformer Official Repository](https://github.com/thuml/iTransformer)

## ⚙️ Configuration

### 🧠 Deconfounding Parameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| Hidden Units | `128` | Number of hidden units in RNN layers |
| Learning Rate | `0.001` | Optimizer learning rate |
| Batch Size | `16` | Samples per training batch |
| Training Epochs | `50` | Number of training iterations |
| RNN Dropout | `0.8` | Dropout probability for regularization |
| Gamma (γ) | `0.6` | Autoregressive process strength |

### 🎯 Correction (iTransformer) Parameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| History Length | `36` | Input sequence length |
| Prediction Length | `3` | Output sequence length |
| Input Dimensions | `55` | Number of input features |
| Batch Size | `8` | Samples per training batch |
| Learning Rate | `0.0001` | Optimizer learning rate |
| Random Seed | `2024` | Reproducibility seed |

## 📈 Results and Performance

Our method demonstrates superior performance in climate model bias correction, particularly for precipitation data in South Australia. Detailed results and comparisons with baseline methods are available in our IJCAI 2025 paper.

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@misc{gao2025deconfoundingmulticauselatentconfounders,
      title={Deconfounding Multi-Cause Latent Confounders: A Factor-Model Approach to Climate Model Bias Correction}, 
      author={Wentao Gao and Jiuyong Li and Debo Cheng and Lin Liu and Jixue Liu and Thuc Duy Le and Xiaojing Du and Xiongren Chen and Yanchang Zhao and Yun Chen},
      year={2025},
      eprint={2408.12063},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2408.12063}, 
}
```

## 📞 Contact

For questions, issues, or collaborations, please:

- 📧 Open an issue on this repository
- 📬 Contact the authors directly
- 🌐 Visit our research group webpage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- iTransformer team for the excellent transformer implementation
- NOAA and IPSL for providing high-quality climate datasets
- The causal inference research community for theoretical foundations

---

<div align="center">
  <p>⭐ If you find this repository helpful, please give it a star! ⭐</p>
</div>
