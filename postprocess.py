import cv2
import numpy as np
from scipy import ndimage


def flood_fill(test_array, h_max=255):
    """
    fill in the hole
    """
    input_array = np.copy(test_array)
    el = ndimage.generate_binary_structure(2, 2).astype(np.int32)
    inside_mask = ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = ndimage.generate_binary_structure(2, 1).astype(np.int32)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(
            input_array,
            ndimage.grey_erosion(
                output_array,
                # size=(3, 3),
                footprint=el,
            ),
        )
    return output_array


def fill_break_line(cw_mask):
    broken_line_h = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    broken_line_h2 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    broken_line_v = np.transpose(broken_line_h)
    broken_line_v2 = np.transpose(broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_h2)
    cw_mask = cv2.morphologyEx(cw_mask, cv2.MORPH_CLOSE, broken_line_v2)

    return cw_mask


def refine_room_region(cw_mask, rm_ind):
    label_rm, num_label = ndimage.label((1 - cw_mask))
    new_rm_ind = np.zeros(rm_ind.shape)
    for j in range(1, num_label + 1):
        mask = (label_rm == j).astype(np.uint8)
        ys, xs = np.where(mask != 0)
        area = (np.amax(xs) - np.amin(xs)) * (np.amax(ys) - np.amin(ys))
        if area < 100:
            continue
        else:
            room_types, type_counts = np.unique(mask * rm_ind, return_counts=True)
            if len(room_types) > 1:
                room_types = room_types[1:]  # ignore background type which is zero
                type_counts = type_counts[1:]  # ignore background count
            new_rm_ind += mask * room_types[np.argmax(type_counts)]

    return new_rm_ind


def fill_holes(rgb_full_img):
    rgb_full = rgb_full_img.copy()
    binary_full = (rgb_full == 0).all(2)
    hole_locations = (
        (ndimage.binary_fill_holes(~binary_full) * 1 - 1 * (~binary_full)).astype(
            np.uint8
        )
    ).nonzero()
    min_same_count = 2
    prev_count = float("inf")
    while len(hole_locations[0]) > 0:
        for x, y in zip(hole_locations[0], hole_locations[1]):
            neighbors = np.array(
                [
                    rgb_full[x + i, y + j, :]
                    for i, j in ((1, 0), (0, 1), (0, -1), (-1, 0))
                ]
            )
            neighbors = neighbors[~(neighbors == 0).all(1)]
            values, counts = np.unique(neighbors, return_counts=True, axis=0)
            if len(counts) == 0:
                continue
            arg_max = np.argmax(counts)
            if counts[arg_max] >= min_same_count:
                rgb_full[x, y] = values[arg_max]
        binary_full = (rgb_full == 0).all(2)
        hole_locations = (
            (ndimage.binary_fill_holes(~binary_full) * 1 - 1 * (~binary_full)).astype(
                np.uint8
            )
        ).nonzero()
        if len(hole_locations[0]) == prev_count:
            min_same_count = 1
        prev_count = len(hole_locations[0])
    return rgb_full.copy()


def clean_room(rm_ind, bd_ind):
    hard_c = (bd_ind > 0).astype(np.uint8)
    # region from room prediction
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask, rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask * new_rm_ind

    return new_rm_ind
