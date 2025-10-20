"""
PGADA: Complete Experiment Pipeline

This script runs the full PGADA experiment pipeline including:
1. Training the model with perturbation-guided adversarial alignment
2. Evaluating the trained model on the test set

The experiment configuration is controlled through files in the configs/ directory.
All outputs (logs, model checkpoints, results) are saved to the directory specified
in configs/experiment_config.py.

Usage:
    python -m scripts.run_experiment

Environment Variables:
    CUDA_VISIBLE_DEVICES: Specify GPU devices (e.g., "0,1")
    
Example:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_experiment
"""

import sys
import warnings
from pathlib import Path

import torch
from loguru import logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the code directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.running_steps import (
    train_model,
    eval_model,
    set_and_print_random_seed,
    prepare_output,
)
from configs import experiment_config


def main():
    """
    Run the complete PGADA experiment pipeline.
    
    This function orchestrates the entire experiment workflow:
    1. Prepares output directories and logging
    2. Sets random seeds for reproducibility
    3. Trains the model using the configured parameters
    4. Evaluates the trained model on the test set
    
    Returns:
        float: Final test accuracy
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting PGADA Experiment Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Prepare output directories and logging
        logger.info("Step 1/4: Preparing output directories...")
        prepare_output()
        
        # Step 2: Set random seeds for reproducibility
        logger.info("Step 2/4: Setting random seeds for reproducibility...")
        random_seed = set_and_print_random_seed()
        
        # Step 3: Train the model
        logger.info("Step 3/4: Training the model...")
        trained_model = train_model()
        
        # Clear GPU cache after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache after training")
        
        # Step 4: Evaluate the model
        logger.info("Step 4/4: Evaluating the trained model...")
        # Reset random seed for consistent evaluation
        set_and_print_random_seed()
        final_accuracy = eval_model(trained_model)
        
        logger.info("=" * 60)
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Final test accuracy: {final_accuracy:.4f}")
        logger.info(f"Results saved in: {experiment_config.SAVE_DIR}")
        logger.info("=" * 60)
        
        return final_accuracy
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        logger.error("Check the logs above for more details")
        raise


if __name__ == "__main__":
    main()
