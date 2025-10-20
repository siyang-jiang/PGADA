"""
Core Training and Evaluation Pipeline for PGADA

This module contains the main functions for training and evaluating models
in the PGADA framework. It handles:
- Model training with episodic learning
- Model evaluation and statistics collection
- Experiment setup and output management
- Model loading and state management

Key Functions:
    - prepare_output(): Sets up experiment directories and logging
    - train_model(): Trains the PGADA model using episodic learning
    - eval_model(): Evaluates trained models on test data
    - load_model(): Loads pre-trained models with proper state handling
"""

import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict
from distutils.dir_util import copy_tree
from pathlib import Path
import random
from shutil import rmtree
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

import configs.evaluation_config
import configs.training_config
from configs import (
    dataset_config,
    training_config,
    model_config,
    experiment_config,
)
from src.utils import set_device, elucidate_ids, get_episodic_loader


def prepare_output() -> None:
    """
    Prepare output directories and logging for the experiment.
    
    Creates the experiment output directory, sets up logging, and saves
    configuration files for reproducibility. If OVERWRITE is True and
    the directory exists, it will be cleared first.
    
    Raises:
        OSError: If directory creation fails
    """
    if not experiment_config.SAVE_RESULTS:
        logger.info("This experiment will not be saved on disk.")
        return
    
    # Handle existing directories
    if experiment_config.SAVE_DIR.exists():
        if experiment_config.OVERWRITE:
            rmtree(str(experiment_config.SAVE_DIR))
            logger.info(
                "Deleted previous content of {directory}",
                directory=experiment_config.SAVE_DIR,
            )
        elif not experiment_config.USE_POLYAXON:
            raise FileExistsError(
                f"Output directory {experiment_config.SAVE_DIR} already exists. "
                "Set OVERWRITE=True in experiment_config.py to overwrite."
            )
    
    # Create output directory
    experiment_config.SAVE_DIR.mkdir(
        parents=True, exist_ok=experiment_config.USE_POLYAXON
    )
    
    # Set up logging to file
    logger.add(experiment_config.SAVE_DIR / "experiment.log")
    
    # Save configuration files for reproducibility
    copy_tree("configs", str(experiment_config.SAVE_DIR / "experiment_configs"))
    
    logger.info(
        "Experiment setup complete. Outputs will be saved to: {directory}",
        directory=experiment_config.SAVE_DIR,
    )


def set_and_print_random_seed() -> int:
    """
    Set random seeds for reproducible experiments.
    
    Sets seeds for numpy, torch, and Python's random module to ensure
    reproducible results across runs. Uses the seed from experiment_config
    or generates a random one if not specified.
    
    Returns:
        int: The random seed that was set
        
    Note:
        Also sets deterministic CUDNN behavior for full reproducibility
    """
    random_seed = experiment_config.RANDOM_SEED
    if random_seed is None:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    
    # Set all random seeds
    np.random.seed(random_seed)
    torch.manual_seed(np.random.randint(0, 2 ** 32 - 1))
    random.seed(np.random.randint(0, 2 ** 32 - 1))
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to: {random_seed}")
    return random_seed


def create_data_loaders() -> Tuple:
    """
    Create episodic data loaders for training, validation, and testing.
    
    Returns:
        Tuple containing (train_loader, val_loader, test_loader, test_dataset)
        test_loader and test_dataset may be None if test validation is disabled
    """
    logger.info("Creating episodic data loaders...")
    
    # Training loader
    train_loader, _ = get_episodic_loader(
        split="train",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_EPISODES,
    )
    
    # Validation loader
    val_loader, _ = get_episodic_loader(
        split="val",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_VAL_TASKS,
    )
    
    # Test loader (optional, for validation during training)
    test_loader = None
    if training_config.TEST_SET_VALIDATION_FREQUENCY:
        test_loader, _ = get_episodic_loader(
            split="test",
            n_way=training_config.N_WAY,
            n_source=training_config.N_SOURCE,
            n_target=training_config.N_TARGET,
            n_episodes=training_config.N_VAL_TASKS,
        )
    
    logger.info("Data loaders created successfully")
    return train_loader, val_loader, test_loader


def train_model() -> nn.Module:
    """
    Train the PGADA model using episodic learning.
    
    Performs the complete training loop including:
    - Data loader initialization
    - Model and optimizer setup
    - Training and validation loops
    - Model checkpointing
    - TensorBoard logging
    
    Returns:
        nn.Module: The trained model with best validation performance
        
    Raises:
        RuntimeError: If training fails
    """
    try:
        # Initialize data loaders
        train_loader, val_loader, test_loader = create_data_loaders()
        
        # Initialize model and optimizer
        logger.info("Initializing model and optimizer...")
        model = set_device(model_config.MODEL(model_config.BACKBONE))
        optimizer = training_config.OPTIMIZER(model.parameters())
        
        # Training state tracking
        best_accuracy = -1.0
        best_epoch = -1
        best_model_state = None
        
        # Set up TensorBoard logging
        writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
        
        logger.info(f"Starting training for {training_config.N_EPOCHS} epochs...")
        
        # Main training loop
        for epoch in range(training_config.N_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{training_config.N_EPOCHS}")
            
            # Training phase
            model.train()
            train_loss, train_acc = model.train_loop(epoch, train_loader, optimizer)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_loss, val_acc, _ = model.eval_loop(val_loader)
            
            # Log metrics
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Train/Accuracy", train_acc, epoch)
            writer.add_scalar("Validation/Loss", val_loss, epoch)
            writer.add_scalar("Validation/Accuracy", val_acc, epoch)
            
            logger.info(
                f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}, Train Loss: {train_loss:.4f}"
            )
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, experiment_config.SAVE_DIR / "best_model.tar")
                logger.info(f"New best model saved (accuracy: {val_acc:.4f})")
            
            # Optional test set validation during training
            if (test_loader is not None and 
                training_config.TEST_SET_VALIDATION_FREQUENCY and
                (epoch + 1) % training_config.TEST_SET_VALIDATION_FREQUENCY == 0):
                
                logger.info("Evaluating on test set...")
                with torch.no_grad():
                    _, test_acc, _ = model.eval_loop(test_loader)
                writer.add_scalar("Test/Accuracy", test_acc, epoch)
                logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Load best model
        logger.info(f"Training completed. Loading best model from epoch {best_epoch + 1}")
        model.load_state_dict(best_model_state)
        logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
        
        writer.close()
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def load_model_episodic(model: nn.Module, state_dict: OrderedDict) -> nn.Module:
    """
    Load state dict for episodic models.
    
    Args:
        model: The model to load state into
        state_dict: The state dictionary to load
        
    Returns:
        nn.Module: Model with loaded state
    """
    model.load_state_dict(state_dict)
    return model


def load_model_non_episodic(
    model: nn.Module, state_dict: OrderedDict, use_fc: bool
) -> nn.Module:
    """
    Load state dict for non-episodic models with proper layer mapping.
    
    Handles the conversion between different model architectures and
    properly maps classifier layers when needed.
    
    Args:
        model: The model to load state into
        state_dict: The state dictionary to load
        use_fc: Whether to use the fully connected classifier layer
        
    Returns:
        nn.Module: Model with loaded state
    """
    if use_fc:
        # Set up the classifier layer
        model.feature.trunk.fc = set_device(
            nn.Linear(
                model.feature.final_feat_dim,
                dataset_config.CLASSES["train"] + dataset_config.CLASSES["val"],
            )
        )
    
    # Handle key name changes for compatibility
    state_keys = list(state_dict.keys())
    for key in state_keys:
        if "clf." in key:
            newkey = key.replace("clf.", "trunk.fc.")
            state_dict[newkey] = state_dict.pop(key)
    
    # Filter and load state dict
    if use_fc:
        filtered_state = OrderedDict(
            [(k, v) for k, v in state_dict.items() if "H." not in k]
        )
    else:
        filtered_state = OrderedDict(
            [(k, v) for k, v in state_dict.items() 
             if "fc" not in k and "H." not in k]
        )
    
    model.feature.load_state_dict(filtered_state)
    return model


def load_model(
    state_path: Path, 
    episodic: bool = True, 
    use_fc: bool = False, 
    force_ot: bool = False
) -> nn.Module:
    """
    Load a pre-trained model from a checkpoint file.
    
    Args:
        state_path: Path to the model checkpoint
        episodic: Whether to load as episodic model
        use_fc: Whether to use fully connected layers
        force_ot: Whether to force optimal transport module
        
    Returns:
        nn.Module: The loaded model
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    if not state_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {state_path}")
    
    # Initialize model
    model = set_device(model_config.MODEL(model_config.BACKBONE))
    
    # Add optimal transport module if forced
    if force_ot:
        model.transportation_module = model_config.TRANSPORTATION_MODULE
        logger.info("Added Optimal Transport module to model")
    
    # Load state dictionary
    try:
        state_dict = torch.load(state_path, map_location='cpu')
        
        # Load model based on type
        if episodic:
            model = load_model_episodic(model, state_dict)
        else:
            model = load_model_non_episodic(model, state_dict, use_fc)
        
        logger.info(f"Successfully loaded model from: {state_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {state_path}: {str(e)}")
        raise


def eval_model(model: nn.Module) -> float:
    """
    Evaluate the trained model on the test set.
    
    Performs comprehensive evaluation including accuracy computation
    and detailed statistics collection. Saves results to CSV and
    logs them to TensorBoard.
    
    Args:
        model: The trained model to evaluate
        
    Returns:
        float: Test set accuracy
        
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        logger.info("Preparing test data for evaluation...")
        
        # Create test data loader
        test_loader, test_dataset = get_episodic_loader(
            split="test",
            n_way=configs.evaluation_config.N_WAY_EVAL,
            n_source=configs.evaluation_config.N_SOURCE_EVAL,
            n_target=configs.evaluation_config.N_TARGET_EVAL,
            n_episodes=configs.evaluation_config.N_TASKS_EVAL,
        )
        
        logger.info("Starting model evaluation...")
        model.eval()
        
        # Perform evaluation
        with torch.no_grad():
            _, accuracy, stats_df = model.eval_loop(test_loader)
        
        # Process and save detailed statistics
        stats_df = elucidate_ids(stats_df, test_dataset)
        stats_df.to_csv(
            experiment_config.SAVE_DIR / "evaluation_statistics.csv", 
            index=False
        )
        
        # Log to TensorBoard
        writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
        writer.add_scalar("Final_Evaluation/Accuracy", accuracy)
        writer.close()
        
        logger.info(f"Evaluation completed. Test accuracy: {accuracy:.4f}")
        logger.info(f"Detailed statistics saved to: evaluation_statistics.csv")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise