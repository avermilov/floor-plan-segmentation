import argparse
import logging
import os
from pathlib import Path

import albumentations as alb
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from dataset import FloorPlanDataset
from net import DFPmodel

PREDICTIONS_DIR = Path("predictions")
MODELS_DIR = Path("models")


def inference(args):
    logging.getLogger("PIL").setLevel(logging.WARNING)  # due to bug in PIL
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(module)s - %(levelname)s"
        + "- %(funcName)s: %(lineno)d - %(message)s",
    )
    os.system("dvc pull --remote drive")

    remap_room = {
        "closet": 5,
        "bathroom": 2,
        "hall": 1,
        "balcony": 4,
        "room": 6,
        "utility": 3,
        "openingtohall": 0,
        "openingtoroom": 0,
    }
    remap_boundary = {
        "utility": 0,
        "openingtohall": 4,
        "openingtoroom": 4,
    }
    tfms = alb.Compose([alb.Resize(512, 512, 0)])
    floorplan_ds = FloorPlanDataset(
        root_dir=args.src_dir,
        num_boundary=args.boundary_channels,
        num_room=args.room_channels,
        remap_room=remap_room,
        remap_boundary=remap_boundary,
        name="inference",
        transform=tfms,
        fill_type="center",
        crop_type="full",
    )
    floorplan_dl = DataLoader(floorplan_ds, batch_size=1, shuffle=False)

    device = args.device
    logging.info(f"Using device: {device}")
    logging.info(f"Loading weights from: {args.weights_path}")
    model = DFPmodel(
        room_channels=args.room_channels, boundary_channels=args.boundary_channels
    )
    model.load_state_dict(
        torch.load(MODELS_DIR / args.weights_path, map_location=args.device)
    )
    model.to(device)
    model.eval()
    logging.info(
        f"Instantiated model with room_channels = {args.room_channels},"
        + f" boundary_channels = {args.boundary_channels}."
    )

    logging.info(f"Found {len(floorplan_ds)} images in src_dir: {args.src_dir}")
    predictions_list = []
    PixelAccRoom = MulticlassAccuracy(
        num_classes=args.room_channels, average="micro"
    ).to(device)
    PixelAccCW = MulticlassAccuracy(
        num_classes=args.boundary_channels, average="micro"
    ).to(device)
    ClassAccRoom = MulticlassAccuracy(
        num_classes=args.room_channels, average="macro"
    ).to(device)
    ClassAccCW = MulticlassAccuracy(
        num_classes=args.boundary_channels, average="macro"
    ).to(device)
    with torch.inference_mode():
        for im, cw, r in tqdm.tqdm(floorplan_dl):
            im, cw, r = im.to(device), cw.to(device), r.to(device)

            logits_r, logits_cw = model(im)

            pixel_acc_room = PixelAccRoom(logits_r, r.argmax(dim=1)).item()
            pixel_acc_cw = PixelAccCW(logits_cw, cw.argmax(dim=1)).item()
            class_acc_room = ClassAccRoom(logits_r, r.argmax(dim=1)).item()
            class_acc_cw = ClassAccCW(logits_cw, cw.argmax(dim=1)).item()

            predictions_list.append(
                {
                    "pixel_acc_room": pixel_acc_room,
                    "pixel_acc_cw": pixel_acc_cw,
                    "class_acc_room": class_acc_room,
                    "class_acc_cw": class_acc_cw,
                }
            )

    predictions_df = pd.DataFrame(predictions_list)
    dst_path = PREDICTIONS_DIR / Path(args.weights_path).parent
    os.makedirs(dst_path, exist_ok=True)
    file_name = "predictions_" + args.src_dir.replace("/", "_") + ".csv"
    csv_path = dst_path / file_name
    predictions_df.to_csv(csv_path, index=False)
    os.system(f"dvc add {csv_path}")
    os.system(f"dvc push {csv_path}.dvc")
    logging.info("Finished inference")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights_path", type=str, default="BEST_EXP/train_20231221_195147/final.pt"
    )
    p.add_argument("--room_channels", type=int, default=7)
    p.add_argument("--boundary_channels", type=int, default=5)
    p.add_argument("--src_dir", type=str, default="data/FloorPlansRussia/val_rare")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    inference(args)
