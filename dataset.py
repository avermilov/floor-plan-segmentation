import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
from PIL import Image
from torch.utils.data import Dataset

boundary_type_to_mask = {
    "background": 0,
    "wall": 1,
    "window": 2,
    "door": 3,
    "utility": 4,
    "openingtohall": 5,
    "openingtoroom": 6,
}
room_type_to_mask = {
    "background": 0,
    "closet": 1,
    "bathroom": 2,
    "hall": 3,
    "balcony": 4,
    "room": 5,
    "utility": 6,
    "openingtohall": 7,
    "openingtoroom": 8,
}
type_to_mask = {
    "defaultwall": 0,
    "openingtohall": 1,
    "openingtoroom": 2,
    "closet": 3,
    "bathroom": 4,
    "hall": 5,
    "balcony": 6,
    "window": 7,
    "background": 8,
    "room": 9,
    "wall": 10,
    "opening": 11,
    "door": 12,
    "utility": 13,
}
mask_to_type = {v: k for k, v in type_to_mask.items()}


def dynamic_crop(image, boundary, room):
    # crop squarely
    while True:
        if (
            np.unique(
                np.concatenate(
                    [
                        image[0, :, :].reshape(-1, 3),
                        image[-1, :, :].reshape(-1, 3),
                        image[:, 0, :].reshape(-1, 3),
                        image[:, -1, :].reshape(-1, 3),
                    ]
                ),
                axis=0,
            ).shape[0]
            == 1
        ):
            image = image[1:-1, 1:-1, :]
            boundary = boundary[1:-1, 1:-1]
            room = room[1:-1, 1:-1]
        else:
            break

    # crop longer side
    height, width, channels = image.shape
    while True:
        if height > width:
            if (
                np.unique(
                    np.concatenate(
                        [image[0, :, :].reshape(-1, 3), image[-1, :, :].reshape(-1, 3)]
                    ),
                    axis=0,
                ).shape[0]
                == 1
            ):
                image = image[1:-1, :, :]
                boundary = boundary[1:-1, :]
                room = room[1:-1, :]
            else:
                break
        elif width > height:
            if (
                np.unique(
                    np.concatenate(
                        [image[:, 0, :].reshape(-1, 3), image[:, -1, :].reshape(-1, 3)]
                    ),
                    axis=0,
                ).shape[0]
                == 1
            ):
                image = image[:, 1:-1, :]
                boundary = boundary[:, 1:-1]
                room = room[:, 1:-1]
            else:
                break
        else:
            break

    return image, boundary, room


def localize(image, boundary, room, pixel_mean_thr: int = 240):
    while True:
        perimeter = np.concatenate(
            [image[0, :, :].reshape(-1, 3), image[-1, :, :].reshape(-1, 3)]
        )
        if (perimeter.mean(axis=1) >= pixel_mean_thr).all():
            image = image[1:-1, :, :]
            boundary = boundary[1:-1, :]
            room = room[1:-1, :]
        else:
            break
    while True:
        perimeter = np.concatenate(
            [image[:, 0, :].reshape(-1, 3), image[:, -1, :].reshape(-1, 3)]
        )
        if (perimeter.mean(axis=1) >= pixel_mean_thr).all():
            image = image[:, 1:-1, :]
            boundary = boundary[:, 1:-1]
            room = room[:, 1:-1]
        else:
            break

    return image, boundary, room


class FillShape(nn.Module):
    def __init__(self, type: str):
        super().__init__()
        self.type = type.strip().lower()
        assert self.type in ["none", "random", "center"]

    def forward(self, image, boundary, room):
        if self.type == "none":
            return image, boundary, room

        height, width, channels = image.shape
        if height == width:
            return image, boundary, room

        if height > width:
            pad_h_left, pad_h_right = 0, 0
            pad_w_left = (
                (height - width) // 2
                if self.type == "center"
                else np.random.randint(0, height - width + 1)
            )
            pad_w_right = height - width - pad_w_left
        else:
            pad_w_left, pad_w_right = 0, 0
            pad_h_left = (
                (width - height) // 2
                if self.type == "center"
                else np.random.randint(0, width - height + 1)
            )
            pad_h_right = width - height - pad_h_left

        padded_room = np.pad(
            room, ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right)), "constant"
        )
        padded_boundary = np.pad(
            boundary, ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right)), "constant"
        )
        padded_image = np.pad(
            image,
            ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right), (0, 0)),
            "constant",
            constant_values=255,
        )

        return padded_image, padded_boundary, padded_room


def remap_values(arr, remap_dict):
    if remap_dict is None or len(remap_dict) == 0:
        return arr

    new_arr = arr.copy()
    for old_value, new_value in remap_dict.items():
        new_arr[arr == old_value] = new_value

    return new_arr


class FloorPlanDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_boundary: int,
        num_room: int,
        remap_room,
        remap_boundary,
        name: str,
        transform=None,
        fill_type: str = "none",
        crop_type: str = "none",
    ):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.num_boundary = num_boundary
        self.num_room = num_room
        self.remap_boundary = {
            boundary_type_to_mask[old]: new for old, new in remap_boundary.items()
        }
        self.remap_room = {
            room_type_to_mask[old]: new for old, new in remap_room.items()
        }
        self.name = name

        self.transform = transform
        self.fill_shape_tsfm = FillShape(type=fill_type)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.crop_type = crop_type.lower().strip()
        assert self.crop_type in ["none", "partial", "full"]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = np.asarray(Image.open(image_path))
        room = np.asarray(Image.open(image_path.replace(".jpg", "_room.png")))
        boundary = np.asarray(Image.open(image_path.replace(".jpg", "_boundary.png")))

        if self.crop_type == "partial":
            image, boundary, room = dynamic_crop(image, boundary, room)
        elif self.crop_type == "full":
            image, boundary, room = localize(image, boundary, room)

        room = remap_values(room, self.remap_room)
        boundary = remap_values(boundary, self.remap_boundary)

        image, boundary, room = self.fill_shape_tsfm(image, boundary, room)

        if self.transform:
            tmp = self.transform(image=image, masks=[boundary, room])
            image, (boundary, room) = tmp["image"], tmp["masks"]

        image = self.to_tensor(image.astype(np.float32) / 255.0)
        boundary = self.to_tensor(
            func.one_hot(torch.LongTensor(boundary), self.num_boundary).numpy()
        )
        room = self.to_tensor(
            func.one_hot(torch.LongTensor(room), self.num_room).numpy()
        )

        return image, boundary, room

    def __len__(self):
        return len(self.image_paths)
