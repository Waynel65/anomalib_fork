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
from anomalib.data import MVTec

from anomalib import TaskType
from anomalib.data.image.folder import Folder, FolderDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import EfficientAd

from pathlib import Path

# set dataset root path
# DATASET_ROOT = Path.cwd() / "datasets" / "MVTec"
DATASET_ROOT = Path.cwd().parent.parent / "Ohio_Manual"
MODEL_NAME = "efficient_ad"
MODELS_PATH = Path.cwd() / "models"
GPR_PROC_METHOD = "gpr_window"
# set task type
# TASK = TaskType.SEGMENTATION
TASK = TaskType.SEGMENTATION
NUM_EPOCHS = 100
IMAGE_SIZE = 504



def main():

    # intialize the datamodule
    datamodule = Folder(
        root=DATASET_ROOT,
        normal_dir="Normal",
        abnormal_dir="Anomalies",
        task=TASK,
        mask_dir=DATASET_ROOT / "Anomalies_mask", 
        image_size=256,
        train_batch_size=2,
        eval_batch_size=2,
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
    model = EfficientAd(model_size="small")

    engine = Engine(
        logger=True,
        default_root_dir=MODELS_PATH,
        max_epochs=NUM_EPOCHS,
        devices=1,
        pixel_metrics=["F1Score", "AUROC"],
        task=TASK,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"{MODELS_PATH}/{MODEL_NAME}/{GPR_PROC_METHOD}/weights",
                monitor=None,
                filename="last",
                save_last=True,
                auto_insert_metric_name=False,
            ),
        ],
        max_steps=70000 if MODEL_NAME == "efficient_ad" else -1,
    )
    engine.fit(model=model, datamodule=datamodule)

    # test the model
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )
    print(test_results)
    return

if __name__ == "__main__":
    main()