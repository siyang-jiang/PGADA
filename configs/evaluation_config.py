from configs.training_config import N_WAY, N_SOURCE, N_TARGET

# Parameters for model evaluation
N_WAY_EVAL = N_WAY
N_SOURCE_EVAL = N_SOURCE
N_TARGET_EVAL = N_TARGET
N_TASKS_EVAL = 2000

# Set to False to evaluate on tasks without domain shift between source and target domains
SUPPORT_QUERY_SHIFT = True
