from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from configs import dataset_config
from src.data_tools.utils import episodic_collate_fn



def set_device(x):
    """
    Switch a tensor to GPU if CUDA is available, to CPU otherwise
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device=device)


def plot_episode(support_images, query_images):
    """
    Plot images of an episode, separating support and query images.
    Args:
        support_images (torch.Tensor): tensor of multiple-channel support images
        query_images (torch.Tensor): tensor of multiple-channel query images
    """

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    support_grid = torchvision.utils.make_grid(support_images)
    matplotlib_imshow(support_grid)
    plt.title("support images")
    plt.show()
    query_grid = torchvision.utils.make_grid(query_images)
    plt.title("query images")
    matplotlib_imshow(query_grid)
    plt.show()


def elucidate_ids(df, dataset):
    """
    Retrieves explicit class and domain names in dataset from their integer index,
        and returns modified DataFrame
    Args:
        df (pd.DataFrame): input DataFrame. Must be the same format as the output of AbstractMetaLearner.get_task_perf()
        dataset (Dataset): the dataset
    Returns:
        pd.DataFrame: output DataFrame with explicit class and domain names
    """
    return df.replace(
        {
            "predicted_label": dataset.id_to_class,
            "true_label": dataset.id_to_class,
            "source_domain": dataset.id_to_domain,
            "target_domain": dataset.id_to_domain,
        }
    )


def get_episodic_loader(
    split: str, n_way: int, n_source: int, n_target: int, n_episodes: int
):
    dataset = dataset_config.DATASET(
        dataset_config.DATA_ROOT, split, dataset_config.IMAGE_SIZE[split]
    )
    sampler = dataset.get_sampler()(
        n_way=n_way,
        n_source=n_source,
        n_target=n_target,
        n_episodes=n_episodes,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=12,
            pin_memory=False, # SY changed
            collate_fn=episodic_collate_fn,
        ),
        dataset,
    )
