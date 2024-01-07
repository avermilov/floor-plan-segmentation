import gc
import logging
import os

import hydra
import numpy as np
import torch.optim
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from loss import balanced_entropy, cross_two_tasks_weight
from postprocess import clean_room
from util import bchw2colormap, boundary2rgb, room2rgb, seed_everything
from visualize import compare_full, compare_rb


def log_to_tb(writer, metric_names, metric_fns, pred, gt, epoch, log_prefix):
    for metric_name, metric_fn in zip(metric_names, metric_fns):
        writer.add_scalar(
            log_prefix + metric_name,
            metric_fn(pred, gt),
            global_step=epoch,
        )


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def remap_color_sum_to_combined(arr, d):
    new_arr = arr.copy()
    for k, v in d.items():
        new_arr[arr == k] = v
    return new_arr


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    project_root_dir = os.getcwd()
    os.makedirs(os.path.join(project_root_dir, "models/"), exist_ok=True)
    os.system("dvc pull")
    run_dir = hydra.core.hydra_config.HydraConfig.get()["run"]["dir"]
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir)
    writer = SummaryWriter(log_dir=run_dir + "/logs/", flush_secs=3)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(module)s - %(levelname)s"
        + " - %(funcName)s: %(lineno)remap_dict - %(message)s",
    )

    logging.info(f"Experiment: {cfg.general.name}")
    DEVICE = cfg.general.device
    logging.info(f"Using device: {DEVICE}")

    seed = cfg.general.seed
    logging.info(f"Using seed: {seed}")
    if seed is not None:
        seed_everything(cfg.general.seed)

    model = hydra.utils.instantiate(cfg.model)
    if cfg.general.model_ckpt is not None:
        model.load_state_dict(torch.load(cfg.general.model_ckpt), strict=False)
    logging.info(
        f"# model parameters: {sum(p.numel() for p in model.parameters()) / 1e6}M"
    )
    model.to(DEVICE)

    OptimizerClass = hydra.utils.get_class(cfg.optimizer.optimizer_type)
    optimizer = OptimizerClass(model.parameters(), **cfg.optimizer.params)

    dataset = hydra.utils.instantiate(cfg.dataset.train)

    shuffle_type = cfg.general.shuffle_type
    shuffle_type = (
        shuffle_type.lower().strip() if shuffle_type is not None else shuffle_type
    )
    do_shuffle = shuffle_type == "standard"

    if "train_share" in cfg.dataset:
        logging.info("Splitting dataset into train/val...")
        train_count = int(cfg.dataset.train_share * len(dataset))
        val_count = len(dataset) - train_count
        train_ds, val_ds = torch.utils.data.random_split(
            dataset,
            [train_count, val_count],
            torch.Generator().manual_seed(cfg.dataset.split_seed),
        )
        val_ds_list = [val_ds]
    else:
        logging.info("Loading val dataset(s)...")
        train_ds = dataset
        val_ds_list = hydra.utils.instantiate(cfg.dataset.val)

    sampler = (
        SubsetRandomSampler(indices=list(range(0, len(train_ds))))
        if shuffle_type == "random"
        else None
    )
    train_loader = DataLoader(
        train_ds,
        num_workers=cfg.general.num_workers,
        batch_size=cfg.general.train_batch_size,
        shuffle=do_shuffle,
        sampler=sampler,
    )

    val_loader_list = [
        DataLoader(
            val_ds, shuffle=False, num_workers=cfg.general.num_workers, batch_size=1
        )
        for val_ds in val_ds_list
    ]

    train_ds_size = len(train_loader.dataset)
    total_val_ds_size = sum([len(val_ds) for val_ds in val_ds_list])

    logging.info(f"Train dataset size: {train_ds_size}")
    logging.info(f"Val dataset size(s): {[len(val_ds) for val_ds in val_ds_list]}")

    scheduler = None
    scheduler_batch_wise_step = getattr(cfg.scheduler, "batch_wise_step", False)
    if cfg.scheduler is not None:
        SchedulerClass = hydra.utils.get_class(cfg.scheduler.scheduler_type)
        if scheduler_batch_wise_step:
            scheduler = SchedulerClass(
                optimizer,
                **cfg.scheduler.params,
                epochs=cfg.general.epochs,
                steps_per_epoch=len(train_loader),
            )
        else:
            scheduler = SchedulerClass(optimizer, **cfg.scheduler.params)

    logging.info(f"Using scheduler: {type(scheduler)}")

    # additional loss coefficients
    room_w = cfg.loss.room_w
    boundary_w = cfg.loss.boundary_w

    # metric initializations
    PixelAccRoom = MulticlassAccuracy(
        num_classes=cfg.general.room_channels, average="micro"
    ).to(cfg.general.device)
    PixelAccCW = MulticlassAccuracy(
        num_classes=cfg.general.boundary_channels, average="micro"
    ).to(cfg.general.device)
    ClassAccRoom = MulticlassAccuracy(
        num_classes=cfg.general.room_channels, average="macro"
    ).to(cfg.general.device)
    ClassAccCW = MulticlassAccuracy(
        num_classes=cfg.general.boundary_channels, average="macro"
    ).to(cfg.general.device)
    PixelAcc = MulticlassAccuracy(
        num_classes=cfg.general.boundary_channels + cfg.general.room_channels - 1,
        average="micro",
    )
    ClassAcc = MulticlassAccuracy(
        num_classes=cfg.general.boundary_channels + cfg.general.room_channels - 1,
        average="macro",
    )
    PixelIOU = MulticlassJaccardIndex(
        num_classes=cfg.general.boundary_channels + cfg.general.room_channels - 1,
        average="micro",
    )
    ClassIOU = MulticlassJaccardIndex(
        num_classes=cfg.general.boundary_channels + cfg.general.room_channels - 1,
        average="macro",
    )

    logging.info("Begin training")
    for epoch in range(cfg.general.epochs):
        run_train = {
            "loss": 0.0,
            "loss_room": 0.0,
            "loss_boundary": 0.0,
            "pixel_acc_room": 0.0,
            "pixel_acc_boundary": 0.0,
            "class_acc_room": 0.0,
            "class_acc_boundary": 0.0,
        }

        for idx, (
            im,
            cw,
            room,
        ) in tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Training #{epoch:03}",
        ):
            im, cw, room = im.to(DEVICE), cw.to(DEVICE), room.to(DEVICE)
            # zero gradients
            optimizer.zero_grad()
            model.train()
            # forward
            logits_r, logits_cw = model(im)
            # loss
            loss1 = balanced_entropy(logits_r, room) * room_w
            loss2 = balanced_entropy(logits_cw, cw) * boundary_w
            w1, w2 = cross_two_tasks_weight(room, cw)
            loss = w1 * loss1 + w2 * loss2
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            run_train["loss"] += loss.item()
            run_train["loss_room"] += loss1.item()
            run_train["loss_boundary"] += loss2.item()

            run_train["pixel_acc_room"] += PixelAccRoom(
                logits_r, room.argmax(dim=1)
            ).item()
            run_train["pixel_acc_boundary"] += PixelAccCW(
                logits_cw, cw.argmax(dim=1)
            ).item()
            run_train["class_acc_room"] += ClassAccRoom(
                logits_r, room.argmax(dim=1)
            ).item()
            run_train["class_acc_boundary"] += ClassAccCW(
                logits_cw, cw.argmax(dim=1)
            ).item()

            writer.add_scalar(
                "lr", get_lr(optimizer), global_step=epoch * len(train_loader) + idx
            )

            if scheduler is not None and scheduler_batch_wise_step:
                scheduler.step()

            if idx % cfg.logging.log_train_every_n_epochs == 0:
                train_example = compare_rb(im, room, cw, logits_r, logits_cw)
                writer.add_figure(f"train/image_{idx:03}", train_example, epoch)
        for name, value in run_train.items():
            writer.add_scalar("train/" + name, value / train_ds_size, global_step=epoch)

        if scheduler is not None and not scheduler_batch_wise_step:
            scheduler.step()

        run_val_total = {
            "loss": 0.0,
            "loss_room": 0.0,
            "loss_boundary": 0.0,
            "pixel_acc_room": 0.0,
            "pixel_acc_boundary": 0.0,
            "class_acc_room": 0.0,
            "class_acc_boundary": 0.0,
        }
        total_run_val_gt = np.array([])
        total_run_val_pred = np.array([])
        total_run_val_pred_post = np.array([])
        with torch.inference_mode():
            model.eval()
            for val_loader in val_loader_list:
                val_ds_size = len(val_loader.dataset)
                run_val = {
                    "loss": 0.0,
                    "loss_room": 0.0,
                    "loss_boundary": 0.0,
                    "pixel_acc_room": 0.0,
                    "pixel_acc_boundary": 0.0,
                    "class_acc_room": 0.0,
                    "class_acc_boundary": 0.0,
                }

                ds_name = val_loader.dataset.name
                for idx, (im, cw, room) in tqdm.tqdm(
                    enumerate(val_loader), total=val_ds_size, desc=f"Val {ds_name}"
                ):
                    im, cw, room = im.to(DEVICE), cw.to(DEVICE), room.to(DEVICE)

                    optimizer.zero_grad()
                    # forward
                    logits_r, logits_cw = model(im)
                    # loss
                    loss1 = balanced_entropy(logits_r, room) * room_w
                    loss2 = balanced_entropy(logits_cw, cw) * boundary_w
                    w1, w2 = cross_two_tasks_weight(room, cw)
                    loss = w1 * loss1 + w2 * loss2
                    # ds running statistics
                    run_val["loss"] += loss.item()
                    run_val["loss_room"] += loss1.item()
                    run_val["loss_boundary"] += loss2.item()
                    # total running statistics
                    run_val_total["loss"] += loss.item()
                    run_val_total["loss_room"] += loss1.item()
                    run_val_total["loss_boundary"] += loss2.item()

                    # metrics
                    pixel_acc_room = PixelAccRoom(logits_r, room.argmax(dim=1))
                    pixel_acc_cw = PixelAccCW(logits_cw, cw.argmax(dim=1))
                    class_acc_room = ClassAccRoom(logits_r, room.argmax(dim=1))
                    class_acc_cw = ClassAccCW(logits_cw, cw.argmax(dim=1))

                    run_val["pixel_acc_room"] += pixel_acc_room.item()
                    run_val["pixel_acc_boundary"] += pixel_acc_cw.item()
                    run_val["class_acc_room"] += class_acc_room.item()
                    run_val["class_acc_boundary"] += class_acc_cw.item()

                    run_val_total["pixel_acc_room"] += pixel_acc_room.item()
                    run_val_total["pixel_acc_boundary"] += pixel_acc_cw.item()
                    run_val_total["class_acc_room"] += class_acc_room.item()
                    run_val_total["class_acc_boundary"] += class_acc_cw.item()

                    # metrics combined
                    predboundary = bchw2colormap(logits_cw)
                    predroom = bchw2colormap(logits_r)
                    predroom_post = clean_room(predroom, predboundary)

                    rgb_room_raw = room2rgb(predroom)
                    rgb_room_post = room2rgb(predroom_post)
                    rgb_boundary = boundary2rgb(predboundary)

                    rgb_full = rgb_boundary.copy()
                    rgb_full[rgb_boundary.sum(axis=2) == 0] = rgb_room_raw[
                        rgb_boundary.sum(axis=2) == 0
                    ]

                    rgb_full_post = rgb_boundary.copy()
                    rgb_full_post[rgb_boundary.sum(axis=2) == 0] = rgb_room_post[
                        rgb_boundary.sum(axis=2) == 0
                    ]

                    rgb_full_pred_post_sum = rgb_full_post.sum(axis=2)
                    rgb_full_pred_sum = rgb_full.sum(axis=2)

                    rgb_full_gt = room2rgb(bchw2colormap(room)) + boundary2rgb(
                        bchw2colormap(cw)
                    )
                    rgb_full_gt_sum = rgb_full_gt.sum(axis=2)

                    total_run_val_gt = np.concatenate(
                        [total_run_val_gt, rgb_full_gt_sum.flatten()]
                    )
                    total_run_val_pred = np.concatenate(
                        [total_run_val_pred, rgb_full_pred_sum.flatten()]
                    )
                    total_run_val_pred_post = np.concatenate(
                        [total_run_val_pred_post, rgb_full_pred_post_sum.flatten()]
                    )

                    # log image
                    if idx % cfg.logging.log_val_every_n_epochs == 0:
                        rb_pred = compare_rb(im, room, cw, logits_r, logits_cw)
                        writer.add_figure(
                            f"val_{ds_name}/image_{idx:03}", rb_pred, epoch
                        )
                        full_pred = compare_full(im, rgb_full, rgb_full_gt)
                        writer.add_figure(
                            f"val_{ds_name}_full/image_{idx:03}", full_pred, epoch
                        )
                # log ds metrics
                for name, value in run_val.items():
                    writer.add_scalar(
                        f"val_{ds_name}/{name}", value / val_ds_size, global_step=epoch
                    )

        # log total metrics
        for name, value in run_val_total.items():
            writer.add_scalar(
                f"val/{name}", value / total_val_ds_size, global_step=epoch
            )
        # log total combined metrics
        unique_vals = np.unique(total_run_val_gt)
        remap_dict = dict(zip(unique_vals, list(range(len(unique_vals)))))
        total_run_val_gt = torch.tensor(
            remap_color_sum_to_combined(total_run_val_gt, remap_dict)
        )
        total_run_val_pred = torch.tensor(
            remap_color_sum_to_combined(total_run_val_pred, remap_dict)
        )
        total_run_val_pred_post = torch.tensor(
            remap_color_sum_to_combined(total_run_val_pred_post, remap_dict)
        )

        # log both raw and postprocessed results
        for suffix, pred in [
            ("_post", total_run_val_pred_post),
            ("", total_run_val_pred),
        ]:
            log_to_tb(
                writer,
                metric_names=[
                    "total_acc_pixel" + suffix,
                    "total_acc_class" + suffix,
                    "total_iou_pixel" + suffix,
                    "total_iou_class" + suffix,
                ],
                metric_fns=[PixelAcc, ClassAcc, PixelIOU, ClassIOU],
                pred=pred,
                gt=total_run_val_gt,
                epoch=epoch,
                log_prefix="val/",
            )

        # save model
        val_loss = run_val_total["loss"] / total_val_ds_size
        if (epoch + 1) % cfg.logging.save_every_n_epochs == 0:
            torch.save(
                model.state_dict(),
                ckpt_dir + f"/model_epoch{epoch:03}_loss{val_loss:.0f}.pt",
            )
        if epoch + 1 == cfg.general.epochs:
            final_model_dir = os.path.join(
                project_root_dir, "models/", run_dir[run_dir.find("/") + 1 :]
            )
            final_model_path = os.path.join(final_model_dir, "final.pt")
            os.makedirs(final_model_dir)
            torch.save(model.state_dict(), final_model_path)
            os.system(f"dvc add {final_model_path}")
            os.system(f"dvc push {final_model_path}.dvc")
        gc.collect()

    logging.info("Finish training")


if __name__ == "__main__":
    main()
