"""Inference Dataset."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any

# import albumentations as A  # noqa: N812
from torch.utils.data.dataset import Dataset

# from anomalib.data.utils import get_image_filenames, get_transforms, read_image

def get_image_filenames(path: str | Path) -> list[Path]:
    pass

def get_transforms(image_size: int | tuple[int, int] = (256, 256)) -> Any:
    pass

def read_image(path: str | Path) -> Any:
    pass

class PredictDataset(Dataset):
    """Inference Dataset to perform prediction.

    Args:
        path (str | Path): Path to an image or image-folder.
        transform (A.Compose | None, optional): Albumentations Compose object describing the transforms that are
            applied to the inputs.
        image_size (int | tuple[int, int] | None, optional): Target image size
            to resize the original image. Defaults to None.
    """

    def __init__(
        self,
        path: str | Path,
        transform: A.Compose | None = None,
        image_size: int | tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        self.image_filenames = get_image_filenames(path)

        if transform is None:
            self.transform = get_transforms(image_size=image_size)
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename)
        pre_processed = self.transform(image=image)
        pre_processed["image_path"] = str(image_filename)

        return pre_processed
