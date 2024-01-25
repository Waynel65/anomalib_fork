import numpy as np
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from anomalib.engine import Engine
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from typing import Any

from anomalib import TaskType
from anomalib.data.image.folder import Folder, FolderDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import EfficientAd

from pathlib import Path

# set dataset root path
DATASET_ROOT = Path.cwd().parent / "datasets" / "hazelnut_toy"
# set task type
TASK = TaskType.SEGMENTATION



def main():

    # intialize the datamodule
    datamodule = Folder(
        root=DATASET_ROOT,
        normal_dir="good",
        abnormal_dir="crack",
        task=TASK,
        mask_dir=DATASET_ROOT / "mask" / "crack",
        image_size=256,
        normalization=InputNormalizationMethod.NONE,  # don't apply normalization, as we want to visualize the images
    )
    datamodule.setup()

    # test if the datamodule is working
    i, data = next(enumerate(datamodule.train_dataloader()))
    print(data.keys(), data["image"].shape)

    # Test images
    i, data = next(enumerate(datamodule.test_dataloader()))
    print(data.keys(), data["image"].shape, data["mask"].shape)

    # get the model
    model = EfficientAd()

    # define callbacks 
    callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="pixel_AUROC",
        ),
        EarlyStopping(
            monitor="pixel_AUROC",
            mode="max",
            patience=3,
        ),
    ]

    # start training
    engine = Engine(
        task=TASK,
        callbacks=callbacks,
        pixel_metrics=["AUROC"],
        accelerator="auto",
        devices=1,
        logger=False,
    )
    engine.fit(model=model, datamodule=datamodule)

    # test the model
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )
    print(test_results)

    #TODO: add the inference script

    return

if __name__ == "__main__":
    main()