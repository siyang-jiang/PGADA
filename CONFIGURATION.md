# Configuration Guide for PGADA

This guide explains how to configure PGADA experiments for different scenarios and datasets.

## 📁 Configuration Structure

The `code/configs/` directory contains all experiment configurations:

```
configs/
├── experiment_config.py      # Global experiment settings
├── dataset_config.py         # Dataset selection and parameters
├── model_config.py          # Model architecture settings
├── training_config.py       # Training hyperparameters
├── evaluation_config.py     # Evaluation protocol settings
└── all_datasets_configs/    # Dataset-specific configurations
    ├── cifar_100_c_config.py
    ├── mini_imagenet_c_config.py
    ├── femnist_config.py
    └── ...
```

## 🔧 Quick Configuration

### 1. Basic Experiment Setup

**Step 1**: Set your experiment name and output directory
```python
# In configs/experiment_config.py
SAVE_DIR = Path("outputs/my_pgada_experiment")
RANDOM_SEED = 42  # For reproducibility
```

**Step 2**: Choose your dataset
```python
# In configs/dataset_config.py
from configs.all_datasets_configs.mini_imagenet_c_config import MINI_IMAGENET_C_CONFIG
DATASET_CONFIG = MINI_IMAGENET_C_CONFIG
```

**Step 3**: Configure model backbone
```python
# In configs/model_config.py
BACKBONE = "resnet18"  # Options: resnet18, resnet50, conv4, etc.
```

### 2. Training Configuration

Key parameters in `configs/training_config.py`:

```python
# Few-shot learning setup
N_WAY = 5                    # Number of classes per episode
N_SOURCE = 5                 # Support examples per class
N_TARGET = 15                # Query examples per class

# Training parameters
N_EPOCHS = 100              # Number of training epochs
N_EPISODES = 100            # Episodes per epoch
N_VAL_TASKS = 100           # Validation tasks

# Optimization
LEARNING_RATE = 0.001       # Base learning rate
OPTIMIZER = torch.optim.Adam # Optimizer choice
```

### 3. Evaluation Configuration

Parameters in `configs/evaluation_config.py`:

```python
# Evaluation protocol
N_WAY_EVAL = 5              # Classes per test episode
N_SOURCE_EVAL = 5           # Support examples for testing
N_TARGET_EVAL = 15          # Query examples for testing
N_TASKS_EVAL = 600          # Total evaluation episodes
```

## 🎯 Common Configuration Scenarios

### Scenario 1: CIFAR-100-C Experiments

```python
# dataset_config.py
from configs.all_datasets_configs.cifar_100_c_config import CIFAR_100_C_CONFIG
DATASET_CONFIG = CIFAR_100_C_CONFIG

# model_config.py
BACKBONE = "resnet18"

# training_config.py
N_WAY = 5
N_SOURCE = 5
N_TARGET = 15
N_EPOCHS = 100
LEARNING_RATE = 0.001
```

### Scenario 2: mini-ImageNet-C with Higher Resolution

```python
# dataset_config.py
from configs.all_datasets_configs.mini_imagenet_c_config import MINI_IMAGENET_C_CONFIG
DATASET_CONFIG = MINI_IMAGENET_C_CONFIG

# model_config.py
BACKBONE = "resnet50"  # Larger backbone for higher resolution

# training_config.py
N_EPOCHS = 200         # More epochs for complex dataset
LEARNING_RATE = 0.0005 # Lower learning rate
```

### Scenario 3: Quick Testing/Debugging

```python
# experiment_config.py
SAVE_RESULTS = False   # Don't save outputs
RANDOM_SEED = 1

# training_config.py
N_EPOCHS = 5          # Few epochs for quick testing
N_EPISODES = 10       # Fewer episodes
N_VAL_TASKS = 20      # Fewer validation tasks
```

### Scenario 4: Production Run with Full Evaluation

```python
# experiment_config.py
SAVE_RESULTS = True
OVERWRITE = False      # Prevent accidental overwrites
SAVE_DIR = Path("outputs/production_run_v1")

# evaluation_config.py
N_TASKS_EVAL = 1000   # More evaluation episodes for statistical significance
```

## 📊 Dataset-Specific Configurations

### CIFAR-100-C Configuration
- **Image Size**: 32×32
- **Classes**: 100 total, split across train/val/test
- **Perturbations**: 19 corruption types with 5 severity levels
- **Recommended Backbone**: ResNet-18 or Conv4

### mini-ImageNet-C Configuration  
- **Image Size**: 84×84
- **Classes**: 100 total (64 train, 16 val, 20 test)
- **Perturbations**: Various corruptions simulating real-world conditions
- **Recommended Backbone**: ResNet-18 or ResNet-50

### tiered-ImageNet-C Configuration
- **Image Size**: 84×84  
- **Classes**: 608 total, hierarchically organized
- **Perturbations**: Similar to mini-ImageNet-C
- **Recommended Backbone**: ResNet-50 (larger capacity needed)

### FEMNIST Configuration
- **Image Size**: 28×28
- **Classes**: 62 characters (digits + uppercase + lowercase)
- **Domain Shift**: Different writers/devices
- **Recommended Backbone**: Conv4 or small ResNet

## ⚙️ Advanced Configuration

### Custom Backbone Integration

To add a new backbone:

1. Define your model in `configs/model_config.py`:
```python
def custom_backbone():
    # Your model definition
    return model

BACKBONE_DICT = {
    "custom": custom_backbone,
    # ...existing backbones
}
```

2. Update the model configuration:
```python
BACKBONE = "custom"
```

### Custom Dataset Integration

1. Create dataset configuration in `configs/all_datasets_configs/`:
```python
# my_dataset_config.py
MY_DATASET_CONFIG = {
    "name": "MyDataset",
    "data_path": "/path/to/data",
    "image_size": 84,
    "n_classes": {"train": 64, "val": 16, "test": 20},
    # ...other parameters
}
```

2. Use in main config:
```python
# dataset_config.py
from configs.all_datasets_configs.my_dataset_config import MY_DATASET_CONFIG
DATASET_CONFIG = MY_DATASET_CONFIG
```

### Hyperparameter Sweeps

For systematic hyperparameter exploration:

```python
# Create multiple config files or use environment variables
import os

# training_config.py
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
N_WAY = int(os.getenv("N_WAY", 5))

# Run with different parameters:
# LEARNING_RATE=0.01 N_WAY=10 python -m scripts.run_experiment
```

## 🔍 Configuration Validation

Before running experiments, validate your configuration:

```python
# Add to your experiment script
def validate_config():
    """Validate configuration consistency"""
    assert training_config.N_WAY <= dataset_config.MIN_CLASSES_PER_SPLIT
    assert experiment_config.SAVE_DIR.parent.exists()
    # Add more validation rules
```

## 💡 Best Practices

1. **Experiment Naming**: Use descriptive names for `SAVE_DIR`
   ```python
   SAVE_DIR = Path(f"outputs/{dataset_name}_{backbone}_{timestamp}")
   ```

2. **Configuration Backup**: The system automatically saves configs to `experiment_configs/`

3. **Reproducibility**: Always set `RANDOM_SEED` for reproducible results

4. **Resource Management**: Adjust batch sizes and episode counts based on GPU memory

5. **Incremental Testing**: Start with small configurations before full runs

6. **Documentation**: Add comments explaining non-obvious parameter choices

## 🚨 Common Configuration Issues

### Issue 1: Out of Memory
**Solution**: Reduce batch size, episode count, or model size
```python
N_EPISODES = 50        # Reduce from 100
N_TARGET = 10          # Reduce query examples
```

### Issue 2: Slow Training
**Solution**: Optimize data loading and reduce validation frequency
```python
TEST_SET_VALIDATION_FREQUENCY = 10  # Validate every 10 epochs instead of every epoch
```

### Issue 3: Poor Convergence
**Solution**: Adjust learning rate and training duration
```python
LEARNING_RATE = 0.0001  # Lower learning rate
N_EPOCHS = 200          # More training epochs
```

### Issue 4: Inconsistent Results
**Solution**: Ensure proper random seed setting
```python
RANDOM_SEED = 42        # Fixed seed
# Also ensure deterministic CUDA operations in code
```

## 📖 Configuration Reference

For complete parameter documentation, see the individual config files:
- [`experiment_config.py`](experiment_config.py) - Global settings
- [`dataset_config.py`](dataset_config.py) - Dataset parameters  
- [`model_config.py`](model_config.py) - Model architecture
- [`training_config.py`](training_config.py) - Training hyperparameters
- [`evaluation_config.py`](evaluation_config.py) - Evaluation settings