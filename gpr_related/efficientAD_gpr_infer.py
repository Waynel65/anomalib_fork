import numpy as np
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from anomalib.engine import Engine
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

# from anomalib.data import PredictDataset
from anomalib.data.image.folder import Folder, FolderDataset
# from gpr_inference_dataset import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib import TaskType

from pathlib import Path


DATASET_ROOT = Path.cwd().parent.parent / "Ohio_Manual"
MODEL_NAME = "efficient_ad"
MODELS_PATH = Path.cwd() / "models"
GPR_PROC_METHOD = "gpr_window"
BEST_CHECKPOINT_PATH = MODELS_PATH / MODEL_NAME / "gpr_window" / "weights" / "last.ckpt"
NUM_EPOCHS = 100
TASK = TaskType.SEGMENTATION


def main():
    # inference_dataset = PredictDataset(path=DATASET_ROOT / "Anomalies/Ohio_Section1-Scan067.png", image_size=(256, 256))
    # inference_dataloader = DataLoader(dataset=inference_dataset)

    # load the datamodule again so that we can let engine create the dataloader
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
    model = EfficientAd(model_size="small")
    predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=BEST_CHECKPOINT_PATH)[1]

    print(predictions.keys())
    print(
        f'Image Shape: {predictions["image"].shape},\n'
        f'mask path: {predictions["mask_path"]}, \n'
        'Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
        'Predicted Mask Shape: {predictions["pred_masks"].shape}',
    )

    image_path = predictions["image_path"][1]
    image_size = predictions["image"].shape[-2:]
    image = np.array(Image.open(image_path).resize(image_size))

    anomaly_map = predictions["anomaly_maps"][0]
    anomaly_map = anomaly_map.cpu().numpy().squeeze()
    plt.imshow(anomaly_map)
    plt.savefig(f"visualization/{MODEL_NAME}_{GPR_PROC_METHOD}_anomaly_map1.png")

    return

if __name__ == "__main__":
    # print(Path.home())
    main()