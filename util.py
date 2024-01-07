import numpy as np
from PIL import Image


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def bchw2colormap(tensor, earlyexit=False):
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result, axis=2)
    return result


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    width, height = imgs[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * width, i // cols * height))
    return grid


def hex_to_rgb(hex_str):
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


pred_boundary_type_to_rgb = {
    "background": hex_to_rgb("000000"),
    "wall": hex_to_rgb("FFFFFF"),
    "window": hex_to_rgb("FF3C80"),
    "door": hex_to_rgb("8020FF"),
    "utility": hex_to_rgb("4080E0"),
    "openingtohall": hex_to_rgb("0EF65F"),
    "openingtoroom": hex_to_rgb("417505"),
}
pred_room_type_to_rgb = {
    "background": hex_to_rgb("000000"),
    "closet": hex_to_rgb("C0C0E0"),
    "bathroom": hex_to_rgb("C0FFFF"),
    "hall": hex_to_rgb("FFA060"),
    "balcony": hex_to_rgb("FFE0E0"),
    "room": hex_to_rgb("E0FFC0"),
    "utility": hex_to_rgb("4080E0"),
    "openingtohall": hex_to_rgb("0EF65F"),
    "openingtoroom": hex_to_rgb("417505"),
}

room_int_to_rgb = {
    0: pred_room_type_to_rgb["background"],
    1: pred_room_type_to_rgb["hall"],
    2: pred_room_type_to_rgb["bathroom"],
    3: pred_room_type_to_rgb["utility"],
    4: pred_room_type_to_rgb["balcony"],
    5: pred_room_type_to_rgb["closet"],
    6: pred_room_type_to_rgb["room"],
}

boundary_int_to_rgb = {
    0: pred_boundary_type_to_rgb["background"],
    1: pred_boundary_type_to_rgb["wall"],
    2: pred_boundary_type_to_rgb["window"],
    3: pred_boundary_type_to_rgb["door"],
    4: pred_boundary_type_to_rgb["openingtohall"],
}


def room2rgb(ind_im):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in room_int_to_rgb.items():
        rgb_im[(ind_im == i)] = rgb

    return rgb_im.astype(int)


def boundary2rgb(ind_im):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in boundary_int_to_rgb.items():
        rgb_im[(ind_im == i)] = rgb

    return rgb_im.astype(int)
