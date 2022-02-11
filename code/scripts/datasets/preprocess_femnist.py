from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from loguru import logger

"""
Write the FEMNIST dataset in the form of three .npy files (for train, val and test).
"""


@click.option(
    "--specs-dir",
    help="Directory where the CSV specification files for train, val and test are located",
    type=Path,
    default=Path("configs/dataset_specs/femnist"),
)
@click.option(
    "--save-dir",
    help="Where to save the data files",
    type=Path,
    default=Path("data/femnist"),
)
@click.option(
    "--mode",
    help="Image mode (see https://pillow.readthedocs.io/en/stable/handbook/concepts.html)",
    type=str,
    default="L",
)
@click.option(
    "--size",
    help="Size of square image",
    type=int,
    default=28,
)
@click.command()
def main(specs_dir, save_dir, mode, size):
    for specs_file_path in specs_dir.glob("*.csv"):
        images_path = pd.read_csv(specs_file_path, index_col=0).img_path
        images = [
            np.asarray(Image.open(img_path).convert(mode).resize((size, size))) / 255
            for img_path in images_path
        ]
        output_path = save_dir / f"{specs_file_path.stem}.npy"
        np.save(output_path, np.stack(images))
        logger.info(
            f"Saved {len(images)} images  from {specs_file_path.name} in {str(output_path)}."
        )


if __name__ == "__main__":
    main()
