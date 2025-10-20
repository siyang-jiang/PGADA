"""
Utility Functions for PGADA Framework

This module provides essential utility functions for the PGADA framework including:
- Device management for GPU/CPU operations
- Data loading and episodic sampling
- Visualization utilities for episodes
- Data processing and statistics helpers

Key Functions:
    - set_device(): Moves tensors to appropriate device (GPU/CPU)
    - get_episodic_loader(): Creates data loaders for few-shot learning
    - plot_episode(): Visualizes support and query images
    - elucidate_ids(): Converts numeric IDs to readable names
"""

from typing import Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from configs import dataset_config
from src.data_tools.utils import episodic_collate_fn


def set_device(model_or_tensor, device: Optional[str] = None):
    """
    Move a model or tensor to the appropriate device (GPU if available, CPU otherwise).
    
    This function automatically detects CUDA availability and moves the input
    to the most appropriate device for computation.
    
    Args:
        model_or_tensor: PyTorch model or tensor to move
        device: Specific device to use (optional). If None, auto-detects best device
        
    Returns:
        The input moved to the specified device
        
    Example:
        >>> model = set_device(MyModel())
        >>> tensor = set_device(torch.randn(3, 224, 224))
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    result = model_or_tensor.to(device=device)
    
    # Log device info for models (not for every tensor to avoid spam)
    if hasattr(model_or_tensor, 'parameters'):
        logger.debug(f"Moved model to device: {device}")
    
    return result


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.
    
    Returns:
        dict: Device information including CUDA availability, device count, etc.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
    }
    
    if torch.cuda.is_available():
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()
    
    return info


def plot_episode(
    support_images: torch.Tensor, 
    query_images: torch.Tensor,
    class_names: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize support and query images from a few-shot learning episode.
    
    Creates a grid visualization showing support images (for learning) and
    query images (for testing) in an episodic learning setup.
    
    Args:
        support_images: Tensor of support images [N, C, H, W]
        query_images: Tensor of query images [N, C, H, W]
        class_names: Optional list of class names for labeling
        save_path: Optional path to save the plot
        figsize: Figure size for the plot
        
    Example:
        >>> plot_episode(support_imgs, query_imgs, 
        ...              class_names=['cat', 'dog', 'bird'])
    """
    def _matplotlib_imshow(img_tensor, title: str = ""):
        """Helper function to display tensor as image"""
        # Handle single channel images
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        
        # Convert to numpy and transpose for matplotlib
        npimg = img_tensor.detach().cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if title:
            plt.title(title)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot support images
    plt.sca(ax1)
    support_grid = torchvision.utils.make_grid(
        support_images.cpu(), 
        normalize=True, 
        padding=2
    )
    _matplotlib_imshow(support_grid, "Support Images (for learning)")
    plt.axis('off')
    
    # Plot query images
    plt.sca(ax2)
    query_grid = torchvision.utils.make_grid(
        query_images.cpu(), 
        normalize=True, 
        padding=2
    )
    _matplotlib_imshow(query_grid, "Query Images (for testing)")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Episode visualization saved to: {save_path}")
    
    plt.show()


def elucidate_ids(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """
    Convert numeric IDs to human-readable class and domain names.
    
    Replaces integer indices with their corresponding string names for better
    interpretability of results and analysis.
    
    Args:
        df: DataFrame with numeric IDs (must match AbstractMetaLearner.get_task_perf() format)
        dataset: Dataset object with id_to_class and id_to_domain mappings
        
    Returns:
        pd.DataFrame: DataFrame with readable class and domain names
        
    Raises:
        AttributeError: If dataset doesn't have required ID mappings
        
    Example:
        >>> readable_df = elucidate_ids(results_df, test_dataset)
        >>> print(readable_df['predicted_label'].unique())  # ['cat', 'dog', 'bird']
    """
    if not hasattr(dataset, 'id_to_class'):
        logger.warning("Dataset missing 'id_to_class' mapping. Skipping class name conversion.")
        return df
    
    if not hasattr(dataset, 'id_to_domain'):
        logger.warning("Dataset missing 'id_to_domain' mapping. Skipping domain name conversion.")
        return df
    
    # Create replacement mapping
    replacement_map = {}
    
    # Add class mappings if columns exist
    if 'predicted_label' in df.columns:
        replacement_map['predicted_label'] = dataset.id_to_class
    if 'true_label' in df.columns:
        replacement_map['true_label'] = dataset.id_to_class
    
    # Add domain mappings if columns exist
    if 'source_domain' in df.columns:
        replacement_map['source_domain'] = dataset.id_to_domain
    if 'target_domain' in df.columns:
        replacement_map['target_domain'] = dataset.id_to_domain
    
    # Apply replacements
    result_df = df.replace(replacement_map)
    
    logger.debug(f"Converted {len(replacement_map)} column types from IDs to names")
    return result_df


def get_episodic_loader(
    split: str, 
    n_way: int, 
    n_source: int, 
    n_target: int, 
    n_episodes: int,
    num_workers: int = 8,
    pin_memory: bool = True
) -> Tuple[DataLoader, Dataset]:
    """
    Create an episodic data loader for few-shot learning.
    
    Sets up a data loader that samples episodes (tasks) for few-shot learning,
    where each episode contains n_way classes with n_source support examples
    and n_target query examples per class.
    
    Args:
        split: Dataset split ('train', 'val', or 'test')
        n_way: Number of classes per episode
        n_source: Number of support examples per class
        n_target: Number of query examples per class  
        n_episodes: Number of episodes to sample
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (DataLoader, Dataset)
        
    Raises:
        ValueError: If invalid split or configuration parameters
        
    Example:
        >>> train_loader, train_dataset = get_episodic_loader(
        ...     split='train', n_way=5, n_source=5, n_target=15, n_episodes=100
        ... )
    """
    # Validate parameters
    valid_splits = ['train', 'val', 'test']
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
    
    if n_way <= 0 or n_source <= 0 or n_target <= 0 or n_episodes <= 0:
        raise ValueError("All episode parameters must be positive integers")
    
    # Create dataset
    try:
        dataset = dataset_config.DATASET(
            root=dataset_config.DATA_ROOT,
            split=split,
            image_size=dataset_config.IMAGE_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to create dataset for split '{split}': {e}")
        raise
    
    # Create episodic sampler
    try:
        sampler = dataset.get_sampler()(
            n_way=n_way,
            n_source=n_source,
            n_target=n_target,
            n_episodes=n_episodes,
        )
    except Exception as e:
        logger.error(f"Failed to create episodic sampler: {e}")
        raise
    
    # Adjust num_workers based on platform and availability
    if num_workers > 0:
        try:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
            num_workers = min(num_workers, max_workers)
        except:
            logger.warning("Could not determine CPU count, using single worker")
            num_workers = 0
    
    # Create data loader
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=episodic_collate_fn,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )
    
    logger.info(
        f"Created episodic loader: {split} split, {n_way}-way {n_source}-shot, "
        f"{n_episodes} episodes, {num_workers} workers"
    )
    
    return loader, dataset


def compute_episode_statistics(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> dict:
    """
    Compute detailed statistics for an episode.
    
    Args:
        predictions: Model predictions [N]
        targets: Ground truth labels [N]
        
    Returns:
        dict: Statistics including accuracy, per-class accuracy, etc.
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Overall accuracy
    accuracy = (predictions == targets).mean()
    
    # Per-class accuracy
    unique_classes = np.unique(targets)
    per_class_acc = {}
    for cls in unique_classes:
        mask = targets == cls
        if mask.sum() > 0:
            per_class_acc[int(cls)] = (predictions[mask] == targets[mask]).mean()
    
    return {
        'accuracy': float(accuracy),
        'per_class_accuracy': per_class_acc,
        'num_samples': len(predictions),
        'num_classes': len(unique_classes)
    }


def log_system_info():
    """Log system information for debugging and reproducibility."""
    device_info = get_device_info()
    
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        logger.info(f"CUDA devices: {device_info['device_count']}")
        logger.info(f"Current device: {device_info['device_name']}")
        logger.info(f"Memory allocated: {device_info['memory_allocated'] / 1e9:.2f} GB")
        logger.info(f"Memory reserved: {device_info['memory_reserved'] / 1e9:.2f} GB")
    else:
        logger.info("Using CPU for computation")
    
    logger.info("=" * 30)
