# PGADA: Perturbation-Guided Adversarial Alignment for Few-shot Learning Under the Support-Query Shift

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2205.03817)
[![Paper](https://img.shields.io/badge/Paper-Springer-green)](https://link.springer.com/chapter/10.1007/978-3-031-05933-9_1)
[![Award](https://img.shields.io/badge/Award-Best%20Student%20Paper%20PAKDD%202022-gold.svg)](https://www.pakdd.net/awards.html)

> **PGADA** addresses the challenging problem of support-query shift in few-shot learning by using perturbation-guided adversarial alignment to improve model robustness across different domains.

## 📰 News

- 🔥 **October 2025**: Using AI to modify the README.md
- 🏆 **May 2022**: PGADA receives **Best Student Paper Award** at **PAKDD 2022**
- 🔥 **March 2022**: PGADA accepted by PAKDD 2022

## 🚀 Quick Start

### Prerequisites

- **GPU**: >8GB VRAM (≥24GB recommended for mini-ImageNet experiments)
- **Python**: 3.7+
- **PyTorch**: 1.7+
- **CUDA**: 10.0+

### Installation

1. **Create and activate virtual environment:**
```bash
# Using virtualenv
virtualenv venv --python=python3.7
source venv/bin/activate

# Or using conda
conda create -n pgada python=3.7
conda activate pgada
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install additional system dependencies (for perturbations):**
```bash
# Required for CIFAR-100-C and mini-ImageNet-C perturbations
sudo apt-get install libmagickwand-dev
```

4. **Set Python path (if needed):**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 📊 Datasets

Download and setup datasets following our [Dataset Guide](DATASETS.md).

**Supported Datasets:**
- CIFAR-100-C-FewShot
- mini-ImageNet-C
- tiered-ImageNet-C  
- FEMNIST

## 🔧 Configuration

All experiments are configured through files in the `code/configs/` directory:

- `experiment_config.py` - Global experiment settings (output paths, random seeds)
- `dataset_config.py` - Dataset-specific parameters
- `model_config.py` - Model architecture and backbone settings
- `training_config.py` - Training hyperparameters and optimization settings
- `evaluation_config.py` - Evaluation protocol parameters

**Example configuration workflow:**
```python
# 1. Set your experiment output directory
# In configs/experiment_config.py
SAVE_DIR = Path("outputs/my_experiment")

# 2. Choose your dataset
# In configs/dataset_config.py  
DATASET_CONFIG = MINI_IMAGENET_C_CONFIG

# 3. Configure model and training
# Modify configs/model_config.py and configs/training_config.py as needed
```

## 🏃‍♂️ Running Experiments

### Full Experiment Pipeline
```bash
# Run complete training + evaluation
python -m scripts.run_experiment
```

### Individual Steps

**Training only:**
```bash
python -m scripts.erm_training
```

**Evaluation only:**
```bash
python -m scripts.eval_model
```

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir=outputs/your_experiment_name
```

## 📈 Reproducing Paper Results

See our detailed [Reproduction Guide](REPRODUCING.md) for exact configurations used in the paper.

## 🏗️ Project Structure

```
PGADA/
├── README.md                    # This file
├── DATASETS.md                  # Dataset setup instructions
├── REPRODUCING.md              # Paper reproduction guide
├── requirements.txt            # Python dependencies
└── code/
    ├── configs/                # Experiment configurations
    │   ├── experiment_config.py    # Global settings
    │   ├── dataset_config.py       # Dataset parameters
    │   ├── model_config.py         # Model architecture
    │   ├── training_config.py      # Training hyperparameters
    │   └── evaluation_config.py    # Evaluation settings
    ├── scripts/                # Executable scripts
    │   ├── run_experiment.py       # Full pipeline
    │   ├── erm_training.py         # Training script
    │   └── eval_model.py           # Evaluation script
    ├── src/                    # Core implementation
    │   ├── methods/                # Few-shot learning algorithms
    │   ├── modules/                # Neural network components
    │   ├── data_tools/             # Data processing utilities
    │   └── utils.py                # Common utilities
    └── outputs/                # Experiment results (created during runs)
```

## 🔬 Key Features

- **Domain Adaptation**: Handles support-query shift in few-shot scenarios
- **Perturbation-Guided**: Uses adversarial perturbations for better alignment
- **Multiple Backbones**: Supports various CNN architectures
- **Comprehensive Evaluation**: Detailed statistics and visualizations
- **Reproducible**: Fixed seeds and detailed configuration tracking

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Base framework adapted from [meta-domain-shift](https://github.com/ebennequin/meta-domain-shift)
- Image perturbations from [robustness](https://github.com/hendrycks/robustness)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{jiang2022pgada,
  title={PGADA: Perturbation-Guided Adversarial Alignment for Few-Shot Learning Under the Support-Query Shift},
  author={Jiang, Siyang and Ding, Wei and Chen, Hsi-Wen and Chen, Ming-Syan},
  booktitle={Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022, Chengdu, China, May 16--19, 2022, Proceedings, Part I},
  pages={3--15},
  year={2022},
  organization={Springer}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ❓ FAQ

**Q: I'm getting "module not found" errors**  
A: Make sure you've set the PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)`

**Q: Training is very slow**  
A: Ensure you're using a GPU and have sufficient VRAM. Consider reducing batch size in training config.

**Q: Results don't match the paper**  
A: Check the [reproduction guide](REPRODUCING.md) for exact hyperparameters and ensure you're using the same random seed.

