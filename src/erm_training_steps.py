"""
Steps used in scripts/erm_training.py
"""

# from typing import OrderedDict
from collections import OrderedDict
# from src.modules import backbones

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer
from torch.utils.data import ConcatDataset, random_split, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import dataset_config, erm_training_config, experiment_config, model_config, training_config
from src.utils import set_device, get_episodic_loader
import configs.evaluation_config
from src.NTXentLoss import NTXentLoss
import copy

class projector_SIMCLR(nn.Module):
    '''
        The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    '''
    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def get_few_shot_split(two_stream=False) -> (Dataset, Dataset):
    temp_train_set = dataset_config.DATASET(
        dataset_config.DATA_ROOT, "train", dataset_config.IMAGE_SIZE, two_stream=two_stream
    )
    temp_train_classes = len(temp_train_set.id_to_class)
    temp_val_set = dataset_config.DATASET(
        dataset_config.DATA_ROOT,
        "val",
        dataset_config.IMAGE_SIZE,
        target_transform=lambda label: label + temp_train_classes,
        two_stream=two_stream
    )
    if hasattr(dataset_config.DATASET, "__name__"):
        if dataset_config.DATASET.__name__ == "CIFAR100CMeta":
            label_mapping = {
                v: k
                for k, v in enumerate(
                    list(temp_train_set.id_to_class.keys())
                    + list(temp_val_set.id_to_class.keys())
                )
            }
            temp_train_set.target_transform = (
                temp_val_set.target_transform
            ) = lambda label: label_mapping[label]

    return temp_train_set, temp_val_set


def get_non_few_shot_split(
    temp_train_set: Dataset, temp_val_set: Dataset
) -> (Subset, Subset):
    train_and_val_set = ConcatDataset(
        [
            temp_train_set,
            temp_val_set,
        ]
    )
    n_train_images = int(
        len(train_and_val_set) * erm_training_config.TRAIN_IMAGES_PROPORTION
    )
    return random_split(
        train_and_val_set,
        [n_train_images, len(train_and_val_set) - n_train_images],
        generator=torch.Generator().manual_seed(
            erm_training_config.TRAIN_VAL_SPLIT_RANDOM_SEED
        ),
    )


def get_data(two_stream=False) -> (DataLoader, DataLoader, int): # mix training and validation
    logger.info("Initializing data loaders...")
    
    if dataset_config.DATASET.__name__ == "FEMNIST":
        train_loader, train_set = get_episodic_loader(
            "train",
            n_way=32,
            n_source=1,
            n_target=1,
            n_episodes=200,
        )
        val_loader, val_set = get_episodic_loader(
            "val",
            n_way=training_config.N_WAY,
            n_source=training_config.N_SOURCE,
            n_target=training_config.N_TARGET,
            n_episodes=training_config.N_VAL_TASKS,
        )
        # Assume that train and val classes are entirely disjoints
        n_classes = len(train_set.id_to_class)
    
    else:
        temp_train_set, temp_val_set = get_few_shot_split(two_stream)

        train_set, val_set = get_non_few_shot_split(temp_train_set, temp_val_set)

        train_loader = DataLoader(
            train_set,
            batch_size=erm_training_config.BATCH_SIZE,
            num_workers=erm_training_config.N_WORKERS,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=erm_training_config.BATCH_SIZE,
            num_workers=erm_training_config.N_WORKERS,
        )
        # Assume that train and val classes are entirely disjoints
        n_classes = len(temp_val_set.id_to_class) + len(temp_train_set.id_to_class)

    return train_loader, val_loader, n_classes


def get_model(n_classes: int) -> nn.Module:
    logger.info(f"Initializing {model_config.BACKBONE.__name__}...")

    model = set_device(model_config.BACKBONE())

    model.clf = set_device(nn.Linear(model.final_feat_dim, n_classes))
    model.H = set_device(model_config.H())
    model.clf_SIMCLR = set_device(projector_SIMCLR(model.final_feat_dim, erm_training_config.SIMCLR_projection_dim))

    model.loss_fn = nn.CrossEntropyLoss(reduction='mean') # SY ADD
    model.loss_fn_SIMCLR = NTXentLoss('cuda', erm_training_config.BATCH_SIZE, erm_training_config.SIMCLR_temp, True)

    model.optimizer = erm_training_config.OPTIMIZER(list(model.trunk.parameters()) + list(model.clf.parameters()) + list(model.clf_SIMCLR.parameters()))
    model.optimizer_H = erm_training_config.OPTIMIZER(model.H.parameters())
    return model


def get_n_batches(data_loader: DataLoader, n_images_per_epoch: int) -> int:
    """
    Computes the number of batches in a training epoch from the intended number of seen images.
    """

    return min(n_images_per_epoch // erm_training_config.BATCH_SIZE, len(data_loader))



def train(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
) -> (OrderedDict, int):
    writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
    n_training_batches = get_n_batches(
        train_loader, erm_training_config.N_TRAINING_IMAGES_PER_EPOCH
    )
    n_val_batches = get_n_batches(
        val_loader, erm_training_config.N_VAL_IMAGES_PER_EPOCH
    )

    test_set_temp1 = dataset_config.DATASET(
        dataset_config.DATA_ROOT, "test", dataset_config.IMAGE_SIZE, augmentation=True, two_stream = True, 
    )
    test_set_temp2 = dataset_config.DATASET(
        dataset_config.DATA_ROOT, "test", dataset_config.IMAGE_SIZE, augmentation=True, two_stream = True, SIMCLR_val = True
    )

    ind = torch.randperm(len(test_set_temp1))
    
    test_set_train_ind = ind[:int(0.9*len(ind))]
    test_set_val_ind = ind[int(0.9*len(ind)):]
    
    test_set_train = Subset(test_set_temp1, test_set_train_ind)
    test_set_val = Subset(test_set_temp2, test_set_val_ind)

    test_loader_train = DataLoader(
        test_set_train,
        batch_size=erm_training_config.BATCH_SIZE,
        num_workers=erm_training_config.N_WORKERS,
        shuffle=True, 
    )

    test_loader_val = DataLoader(
        test_set_val,
        batch_size=erm_training_config.BATCH_SIZE,
        num_workers=erm_training_config.N_WORKERS,
        shuffle=False,
    )

    if erm_training_config.batch_validate:
        model.loss_fn_SIMCLR_val = NTXentLoss('cuda', erm_training_config.BATCH_SIZE, erm_training_config.SIMCLR_temp, True)
    else:
        model.loss_fn_SIMCLR_val = NTXentLoss('cuda', len(test_set_val), erm_training_config.SIMCLR_temp, True)

    if erm_training_config.SIMCLR:
        min_val_loss = float("inf")
    else:
        max_val_acc = -float("inf")
    best_model_epoch = 0
    logger.info("Model and data are ready. Starting training...")
    for epoch in range(erm_training_config.N_EPOCHS):
        
        if epoch > best_model_epoch + 10:
            logger.info(f"Training early stops.")
            return

        model, average_loss = training_epoch(
            model, train_loader, epoch, n_training_batches, test_loader_train
        )

        writer.add_scalar("Train/loss", average_loss, epoch)

        if erm_training_config.SIMCLR:
            val_loss = validation(model, val_loader, epoch, n_val_batches, test_loader_val)
            writer.add_scalar("Val/loss", val_loss, epoch)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model_epoch = epoch
                logger.info(f"Best model found at training epoch {best_model_epoch}.")
                state_dict_path = (
                    experiment_config.SAVE_DIR
                    / f"{model_config.BACKBONE.__name__}_{dataset_config.DATASET.__name__ if hasattr(dataset_config.DATASET, '__name__') else dataset_config.DATASET.func.__name__}_{epoch}.tar"
                )
                torch.save(model.state_dict(), state_dict_path)
        else:
            val_acc = validation_acc(model, val_loader, epoch, n_val_batches, test_loader_val)
            writer.add_scalar("Val/acc", val_acc, epoch)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_model_epoch = epoch
                logger.info(f"Best model found at training epoch {best_model_epoch} .")
                state_dict_path = (
                    experiment_config.SAVE_DIR
                    / f"{model_config.BACKBONE.__name__}_{dataset_config.DATASET.__name__ if hasattr(dataset_config.DATASET, '__name__') else dataset_config.DATASET.func.__name__}_{epoch}.tar"
                )
                torch.save(model.state_dict(), state_dict_path)
        
    return

def training_epoch(
    model: nn.Module, data_loader: DataLoader, epoch: int, n_batches: int, test_loader_train: DataLoader
) -> (nn.Module, float):
    loss_clf_list = []
    loss_cos_list = []
    model.train()
    
    if dataset_config.DATASET.__name__ == "FEMNIST":
        for batch_id, (support_images,
                support_labels,
                query_images,
                query_labels,
                class_ids,
                source_domain,
                target_domain,), (test_img1, test_img2, _, _) in zip(range(n_batches), data_loader, test_loader_train):
            labels = torch.as_tensor([class_ids[i] for i in support_labels], dtype=torch.long)
            model, loss_clf, loss_cos = fit(model, set_device(support_images), set_device(query_images), set_device(labels), set_device(test_img1), set_device(test_img2))

            loss_clf_list.append(loss_clf)
            loss_cos_list.append(loss_cos)

            print(f"epoch {epoch} [{batch_id+1:03d}/{n_batches}]: clf loss={np.asarray(loss_clf_list).mean():.3f}, cos loss={np.asarray(loss_cos_list).mean():.3f}", end="     \r")
    else:
        for batch_id, (images, images_perturbation, labels, _), (test_img1, test_img2, _, _) in zip(range(n_batches), data_loader, test_loader_train):
            model, loss_clf, loss_cos = fit(model, set_device(images), set_device(images_perturbation), set_device(labels), set_device(test_img1), set_device(test_img2))

            loss_clf_list.append(loss_clf)
            loss_cos_list.append(loss_cos)

            print(f"epoch {epoch} [{batch_id+1:04d}/{n_batches}]: clf loss={np.asarray(loss_clf_list).mean():.3f}, cos loss={np.asarray(loss_cos_list).mean():.3f}", end="     \r")
    print()                                                                                                                                                                                                                     
    return model, np.asarray(loss_clf_list).mean() + np.asarray(loss_cos_list).mean()

def fit(
    model: nn.Module, images: torch.Tensor, images_perturbation: torch.Tensor, 
    labels: torch.Tensor, test_img1: torch.Tensor, test_img2: torch.Tensor
) -> (nn.Module, float):

    #train H
    model.optimizer_H.zero_grad()
    model.H.eval()
    f_H = model.trunk(model.H(images)) 
    f_perturbation = model.trunk(images_perturbation)
    # f_H_norm = f_H / f_H.norm(dim=1)[:, None]
    # f_perturbation_norm = f_perturbation / f_perturbation.norm(dim=1)[:, None]    
    # loss_cos_similarity = torch.mm(f_H_norm, f_perturbation_norm.transpose(0,1)).mean()
    loss_cos_similarity = F.cosine_similarity(f_H, f_perturbation).mean()
    out = model.clf(f_H)
    loss_clf = model.loss_fn(out, labels)
    loss_H = loss_clf + loss_cos_similarity
    # loss_H = loss_cos_similarity
    loss_H.backward()
    model.optimizer_H.step()
    
    #train M, clf
    model.optimizer.zero_grad()
    model.H.train()
    with torch.no_grad():
        images_H = model.H(images)
    out   = model.clf(model.trunk(images_H))
    out_p = model.clf(model.trunk(images_perturbation))
    loss_CE = model.loss_fn(out, labels) + model.loss_fn(out_p, labels)
    if erm_training_config.SIMCLR:
        z1 = model.clf_SIMCLR(model.trunk(test_img1))
        z2 = model.clf_SIMCLR(model.trunk(test_img2))
        loss_SIMCLR = model.loss_fn_SIMCLR(z1, z2)
        loss_M_clf = loss_CE + loss_SIMCLR
    else:
        loss_M_clf = loss_CE
    loss_M_clf.backward()
    model.optimizer.step()

    return model, loss_M_clf.item(), loss_cos_similarity.item()

def validation(
    model: nn.Module, data_loader: DataLoader, epoch: int, n_batches: int, test_loader_val: DataLoader
) -> float:

    if erm_training_config.batch_validate:
        losses_SIMCLR = []
    else:
        z1s = []
        z2s = []

    model.eval()
    with torch.no_grad():
        for batch_id, (test_img1, test_img2, _, _) in zip(range(n_batches), test_loader_val):
            test_img1 = set_device(test_img1)
            test_img2 = set_device(test_img2)

            z1 = model.clf_SIMCLR(model.trunk(test_img1))
            z2 = model.clf_SIMCLR(model.trunk(test_img2))

            if erm_training_config.batch_validate:
                if len(test_img1) != erm_training_config.BATCH_SIZE:
                    criterion_small_set = NTXentLoss(
                        'cuda', len(test_img1), erm_training_config.SIMCLR_temp, True)
                    losses_SIMCLR.append(criterion_small_set(z1, z2))
                else:
                    losses_SIMCLR.append(model.loss_fn_SIMCLR_val(z1, z2))
            else:
                z1s.append(z1)
                z2s.append(z2)

    if erm_training_config.batch_validate:
        loss_SIMCLR = torch.stack(losses_SIMCLR).mean()
    else:
        z1s = torch.cat(z1s, dim=0)
        z2s = torch.cat(z2s, dim=0)
        loss_SIMCLR = model.loss_fn_SIMCLR_val(z1s, z2s)

    if dataset_config.DATASET.__name__ == "FEMNIST":
        val_model = set_device(model_config.MODEL(model_config.BACKBONE))
        val_model.feature = model
        val_model.eval()
        with torch.no_grad():
            loss, acc, stats_df = val_model.eval_loop(data_loader)
    else:
        with torch.no_grad():
            for batch_id, (images, images_perturbation, labels, _) in zip(range(n_batches), data_loader):
                images_perturbation = set_device(images_perturbation)
                labels = set_device(labels)

                out_p = model.clf(model.trunk(images_perturbation))
                loss = model.loss_fn(out_p, labels)

    print(f"epoch {epoch} : loss={loss+loss_SIMCLR:.3f}")
    return loss + loss_SIMCLR

def validation_acc(model: nn.Module, data_loader: DataLoader, epoch: int, n_batches: int, test_loader_val: DataLoader) -> float:
    if dataset_config.DATASET.__name__ == "FEMNIST":
        val_model = set_device(model_config.MODEL(model_config.BACKBONE))
        val_model.feature = model
        val_model.eval()
        loss, acc, stats_df = val_model.eval_loop(data_loader)
        
        return acc
    else:
        val_acc_list = []
        model.eval()
        with torch.no_grad():
            for batch_id, (images, images_perturbation, labels, _) in zip(range(n_batches), data_loader):
                    val_acc_list.append(
                        float(
                            (
                                model.clf(model.trunk(set_device(images_perturbation))).data.topk(1, 1, True, True)[1][:, 0]
                                == set_device(labels)
                            ).sum()
                        )
                        / len(labels)
                    )
                    print(f"validation [{batch_id+1:03d}/{n_batches}]: acc={np.asarray(val_acc_list).mean():.3f}", end="     \r")
        print()
        return np.asarray(val_acc_list).mean()