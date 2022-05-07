from pathlib import Path

# Global config for the experiment

RANDOM_SEED = 1  # If None, random seed will be randomly sampled
SAVE_RESULTS = True
OVERWRITE = False  # If True, will erase all previous content of SAVE_DIR

# Check this Before experiments
SAVE_DIR = Path("outputs/SHIFT-5way1shot-Matching-val16")

# Additional parameters, only used for polyaxon
USE_POLYAXON = True
SECOND_EVAL_SAVE_DIR = SAVE_DIR / "extra_eval"
