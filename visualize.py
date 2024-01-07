import numpy as np
from matplotlib import pyplot as plt

from postprocess import clean_room
from util import bchw2colormap, boundary2rgb, room2rgb


def compare_rb(images, rooms, boundaries, r, cw):
    image = (bchw2colormap(images, earlyexit=True) * 255).astype(np.uint8)
    room = bchw2colormap(rooms).astype(np.uint8)
    boundary = bchw2colormap(boundaries).astype(np.uint8)
    r = bchw2colormap(r).astype(np.uint8)
    cw = bchw2colormap(cw).astype(np.uint8)
    room_post = clean_room(r, cw).astype(np.uint8)

    f = plt.figure(dpi=170)
    for i, (name, img) in enumerate(
        [
            ("image", image),
            ("room gt", room2rgb(room)),
            ("bd gt", boundary2rgb(boundary)),
            ("r pred post", room2rgb(room_post)),
            ("r pred", room2rgb(r)),
            ("bd pred", boundary2rgb(cw)),
        ],
        start=1,
    ):
        plt.subplot(2, 3, i)
        plt.tick_params(left=False, right=False, top=False, bottom=False)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])

        plt.title(name)
        plt.imshow(img)

    return f


def compare_full(images, pred_full, gt_full):
    image = (bchw2colormap(images, earlyexit=True) * 255).astype(np.uint8)

    f = plt.figure(figsize=(8, 4), dpi=150)
    for i, (name, img) in enumerate(
        [("image", image), ("pred", pred_full), ("gt", gt_full)], start=1
    ):
        plt.subplot(1, 3, i)
        plt.tick_params(left=False, right=False, top=False, bottom=False)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])

        plt.title(name)
        plt.imshow(img)

    return f
