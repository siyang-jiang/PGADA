import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict
from distutils.dir_util import copy_tree
from pathlib import Path
import random
from shutil import rmtree

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


def prepare_output():
    if experiment_config.SAVE_RESULTS:
        if experiment_config.OVERWRITE & experiment_config.SAVE_DIR.exists():
            rmtree(str(experiment_config.SAVE_DIR))
            logger.info(
                "Deleting previous content of {directory}",
                directory=experiment_config.SAVE_DIR,
            )

        experiment_config.SAVE_DIR.mkdir(
            parents=True, exist_ok=experiment_config.USE_POLYAXON
        )
        logger.add(experiment_config.SAVE_DIR / "running.log")
        copy_tree("configs", str(experiment_config.SAVE_DIR / "experiment_parameters"))
        logger.info(
            "Parameters and outputs of this experiment will be saved in {directory}",
            directory=experiment_config.SAVE_DIR,
        )

    else:
        logger.info("This experiment will not be saved on disk.")


def set_and_print_random_seed():
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Returns:
        int: numpy random seed

    """
    random_seed = experiment_config.RANDOM_SEED
    if not random_seed:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)
    torch.manual_seed(np.random.randint(0, 2 ** 32 - 1))
    random.seed(np.random.randint(0, 2 ** 32 - 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed : {random_seed}")

    return random_seed


def train_model():
    logger.info("Initializing data loaders...")
    train_loader, _ = get_episodic_loader(
        "train",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_EPISODES,
    )
    val_loader, _ = get_episodic_loader(
        "val",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_VAL_TASKS,
    )
    if training_config.TEST_SET_VALIDATION_FREQUENCY:
        test_loader, _ = get_episodic_loader(
            "test",
            n_way=training_config.N_WAY,
            n_source=training_config.N_SOURCE,
            n_target=training_config.N_TARGET,
            n_episodes=training_config.N_VAL_TASKS,
        )

    logger.info("Initializing model...")

    model = set_device(model_config.MODEL(model_config.BACKBONE))
    optimizer = training_config.OPTIMIZER(model.parameters())

    max_acc = -1.0
    best_model_epoch = -1
    best_model_state = None

    writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)

    logger.info("Model and data are ready. Starting training...")
    for epoch in range(training_config.N_EPOCHS): #SY: seeing the batch size
        # Set model to training mode
        model.train()
        # Execute a training loop of the model
        train_loss, train_acc = model.train_loop(epoch, train_loader, optimizer)
        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Train/acc", train_acc, epoch)
        # Set model to evaluation mode
        model.eval()
        # Evaluate on validation set
        val_loss, val_acc, _ = model.eval_loop(val_loader)
        writer.add_scalar("Val/loss", val_loss, epoch)
        writer.add_scalar("Val/acc", val_acc, epoch)

        # We make sure the best model is saved on disk, in case the training breaks
        if val_acc > max_acc:
            max_acc = val_acc
            best_model_epoch = epoch
            best_model_state = model.state_dict()
            torch.save(best_model_state, experiment_config.SAVE_DIR / "best_model.tar")

        if training_config.TEST_SET_VALIDATION_FREQUENCY:
            if (
                epoch % training_config.TEST_SET_VALIDATION_FREQUENCY
                == training_config.TEST_SET_VALIDATION_FREQUENCY - 1
            ):
                logger.info("Validating on test set...")
                _, test_acc, _ = model.eval_loop(test_loader)
                writer.add_scalar("Test/acc", test_acc, epoch)

    logger.info(f"Training over after {training_config.N_EPOCHS} epochs")
    logger.info("Retrieving model with best validation accuracy...")
    model.load_state_dict(best_model_state)
    logger.info(f"Retrieved model from epoch {best_model_epoch}")

    writer.close()

    return model


def load_model_episodic(model: nn.Module, state_dict: OrderedDict) -> nn.Module: # episodic 
    model.load_state_dict(state_dict)
    return model


def load_model_non_episodic(
    model: nn.Module, state_dict: OrderedDict, use_fc: bool
) -> nn.Module:
    if use_fc:
        # model.feature.clf = set_device(
        #     nn.Linear(
        #         model.feature.final_feat_dim,
        #         dataset_config.CLASSES["train"] + dataset_config.CLASSES["val"],
        #     )
        # )
        model.feature.trunk.fc = set_device(
            nn.Linear(
                model.feature.final_feat_dim,
                dataset_config.CLASSES["train"] + dataset_config.CLASSES["val"],
            )
        )
    state_keys = list(state_dict.keys())
    for _, key in enumerate(state_keys):
        if "clf." in key:
            newkey = key.replace("clf.", "trunk.fc.")
            state_dict[newkey] = state_dict.pop(key)

    model.feature.load_state_dict(
        OrderedDict([(k, v) for k, v in state_dict.items() if "H." not in k])
        if use_fc
        else OrderedDict([(k, v) for k, v in state_dict.items() if "fc" not in k and "H." not in k])
    )

    # model.feature.load_state_dict(
    #     state_dict
    #     if use_fc
    #     else OrderedDict([(k, v) for k, v in state_dict.items() if ".fc." not in k])
    # )
    return model


def load_model(
    state_path: Path, episodic: bool, use_fc: bool, force_ot: bool
) -> nn.Module:
    model = set_device(model_config.MODEL(model_config.BACKBONE))

    if force_ot:
        model.transportation_module = model_config.TRANSPORTATION_MODULE
        logger.info("Forced the Optimal Transport module into the model.")

    state_dict = torch.load(state_path)
    model = (
        load_model_episodic(model, state_dict)
        if episodic
        else 
        load_model_non_episodic(model, state_dict, use_fc)
    )

    logger.info(f"Loaded model from {state_path}")

    return model


def eval_model(model):
    logger.info("Initializing test data...")
    test_loader, test_dataset = get_episodic_loader(
        "test",
        n_way=configs.evaluation_config.N_WAY_EVAL,
        n_source=configs.evaluation_config.N_SOURCE_EVAL,
        n_target=configs.evaluation_config.N_TARGET_EVAL,
        n_episodes=configs.evaluation_config.N_TASKS_EVAL,
    )

    logger.info("Starting model evaluation...")
    model.eval()

    _, acc, stats_df = model.eval_loop(test_loader)

    stats_df = elucidate_ids(stats_df, test_dataset)

    stats_df.to_csv(experiment_config.SAVE_DIR / "evaluation_stats.csv", index=False)
    writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
    writer.add_scalar("Evaluation accuracy", acc)
    writer.close()
    return acc