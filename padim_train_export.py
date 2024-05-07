from anomalib.data import Folder
from anomalib import TaskType
from pathlib import Path
from anomalib.models import Padim

def main():
    datamodule = Folder(
        name="3",
        root=Path.cwd() / "datasets/3",
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_split_ratio=0.2,
        image_size=(256, 256),
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
        task=TaskType.CLASSIFICATION,
    )
    datamodule.setup()

    i, data = next(enumerate(datamodule.val_dataloader()))
    print(data.keys())

    # Check image size
    print(data["image"].shape)

    

    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )

    from anomalib.engine import Engine
    from anomalib.utils.normalization import NormalizationMethod

    engine = Engine(
        normalization=NormalizationMethod.MIN_MAX,
        threshold="F1AdaptiveThreshold",
        task=TaskType.CLASSIFICATION,
        image_metrics=["AUROC"],
        accelerator="auto",
        check_val_every_n_epoch=1,
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
    )

    engine.fit(model=model, datamodule=datamodule)

    from anomalib.deploy import ExportType

# Exporting model to OpenVINO
    openvino_model_path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=str(Path.cwd()),
    )
    print(openvino_model_path)
if __name__ == "__main__":
    main()