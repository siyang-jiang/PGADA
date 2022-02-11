from pathlib import Path

import click
from loguru import logger
from torchvision import transforms
from tqdm import tqdm

from src.data_tools.datasets import TieredImageNetC


@click.option(
    "--input-dir",
    help="Path to ImageNet data",
    type=Path,
    default=Path("/ILSVRC2015/Data/CLS-LOC/train"),
)
@click.option(
    "--output-dir",
    help="Where to save the tieredImageNet-C dataset",
    type=Path,
    default=Path("/tiered_imagenet_c"),
)
@click.command()
def main(input_dir, output_dir):
    train_set = TieredImageNetC(input_dir, "train", 224, load_corrupted_dataset=False)
    val_set = TieredImageNetC(input_dir, "val", 224, load_corrupted_dataset=False)
    test_set = TieredImageNetC(input_dir, "test", 224, load_corrupted_dataset=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    trans = transforms.ToPILImage()

    def save_set(dataset, setname):
        logger.info(f"Writing {setname} set in {output_dir / setname}...")
        for i, x in tqdm(enumerate(dataset), total=len(dataset)):
            this_output_dir = (
                output_dir
                / setname
                / dataset.id_to_class[x[1]]
                / dataset.id_to_domain[x[2]]
            )
            this_output_dir.mkdir(parents=True, exist_ok=True)
            trans(x[0]).save(this_output_dir / f"{i:08d}.png", format="PNG")

        logger.info(f"{setname} set has been written.")

    save_set(train_set, "train")
    save_set(val_set, "val")
    save_set(test_set, "test")


if __name__ == "__main__":
    main()
