"""
Global Experiment Configuration for PGADA

This module contains the main configuration parameters for PGADA experiments.
These settings control the overall experiment behavior, output management,
and reproducibility settings.

Configuration Categories:
    - Reproducibility: Random seed settings
    - Output Management: Result saving and directory handling
    - Platform Support: Polyaxon integration settings

Usage:
    Modify the values in this file before running experiments to customize
    the experiment behavior. The SAVE_DIR should be updated for each new
    experiment to avoid overwriting previous results.
"""

from pathlib import Path

# =============================================================================
# REPRODUCIBILITY SETTINGS
# =============================================================================

# Random seed for reproducible experiments
# Set to None for random seed generation, or specify an integer for fixed seed
RANDOM_SEED = 1

# =============================================================================
# OUTPUT MANAGEMENT
# =============================================================================

# Whether to save experiment results to disk
# Set to False for testing/debugging runs that don't need to be saved
SAVE_RESULTS = True

# Whether to overwrite existing output directory
# WARNING: Setting this to True will delete all previous content in SAVE_DIR
OVERWRITE = False

# Output directory for experiment results
# This directory will contain:
# - Model checkpoints (best_model.tar)
# - Training logs (experiment.log)
# - Configuration backup (experiment_configs/)
# - TensorBoard logs
# - Evaluation results (evaluation_statistics.csv)
#
# IMPORTANT: Update this path for each new experiment to avoid conflicts
SAVE_DIR = Path("outputs/pgada_experiment")

# =============================================================================
# PLATFORM SETTINGS
# =============================================================================

# Enable Polyaxon platform integration
# Set to True when running on Polyaxon clusters, False for local execution
USE_POLYAXON = False

# Secondary evaluation directory (used with Polyaxon)
# Additional evaluation results may be saved here for extended analysis
SECOND_EVAL_SAVE_DIR = SAVE_DIR / "extended_evaluation"

# =============================================================================
# EXPERIMENT METADATA
# =============================================================================

# Experiment description (optional, for documentation purposes)
EXPERIMENT_DESCRIPTION = """
PGADA experiment with perturbation-guided adversarial alignment.
This experiment trains a few-shot learning model that handles support-query
domain shift using adversarial perturbations for improved generalization.
"""

# Experiment tags (optional, for organization)
EXPERIMENT_TAGS = ["pgada", "few-shot", "domain-adaptation", "adversarial"]
