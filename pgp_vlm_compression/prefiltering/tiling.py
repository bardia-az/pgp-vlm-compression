import numpy as np
from PIL import Image, ImageOps
import math
import cv2

import torch
from torch import Tensor



def preprocess_image_sliding_window(image, clip_preprocess, input_size=224, stride=112):
    tiles, positions, padded_img, scale, pad = tile_image_sliding_window(
        image, tile_size=input_size, stride=stride
    )
    pixel_values_clip = [clip_preprocess(tile) for tile in tiles]
    pixel_values_clip = torch.stack(pixel_values_clip)
    return pixel_values_clip, positions, padded_img, scale, pad


def tile_image_sliding_window(image: Image.Image, tile_size: int = 224, stride: int = 112):
    """
    Split an image into overlapping tiles using a sliding window, automatically upscaling
    and padding to get the desired output size.

    Returns:
        tiles (list of PIL.Image)
        positions (list of (x, y))
        padded_img (PIL.Image)
        grid_shape (num_rows, num_cols)
        scale (float): resize factor applied to the original image
    """
    orig_w, orig_h = image.size

    target_w = math.ceil(orig_w / tile_size) * tile_size
    target_h = math.ceil(orig_h / tile_size) * tile_size

    resized_w = target_w
    resized_h = target_h
    scale = 1.0 # not important
    if (resized_w, resized_h) != (orig_w, orig_h):
        image_resized = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    else:
        image_resized = image

    # pad if necessary to reach the target size
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    pad_right = target_w - resized_w - pad_left
    pad_bottom = target_h - resized_h - pad_top
    
    np_img = np.array(image_resized)
    padded_np = cv2.copyMakeBorder(
        np_img,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_REPLICATE
    )
    padded_img = Image.fromarray(padded_np)

    tiles, positions = [], []
    for y in range(0, target_h - tile_size + 1, stride):
        for x in range(0, target_w - tile_size + 1, stride):
            tile = padded_img.crop((x, y, x + tile_size, y + tile_size))
            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions, padded_img, scale, (pad_left, pad_top, pad_right, pad_bottom)


def merge_score_tiles(tile_scores: torch.Tensor, positions: list, img_size: tuple, tile_size: int, stride: int = 112):
    """
    Merge per-tile scores into a blockwise score map, one value per stride x stride block.

    Args:
        tile_scores (torch.Tensor): (N,) array of scores per tile.
        positions (list): List of (x, y) top-left coords for each tile.
        stride (int): Stride used during tiling.
    
    Returns:
        block_scores: 2D array (num_blocks_y, num_blocks_x)
    """

    # Determine number of blocks in each dimension
    w, h = img_size
    num_blocks_x = w // stride
    num_blocks_y = h // stride
    assert w % stride == 0 and h % stride == 0, "w and h must be multiples of stride"

    # tile covers s x s blocks
    s = tile_size // stride
    assert tile_size % stride == 0, "tile_size must be divisible by stride for exact blockwise merging"

    block_sum = torch.zeros((num_blocks_y, num_blocks_x), dtype=torch.float32, device=tile_scores.device)
    block_count = torch.zeros((num_blocks_y, num_blocks_x), dtype=torch.float32, device=tile_scores.device)

    for (x, y), score in zip(positions, tile_scores):
        bx0 = x // stride
        by0 = y // stride
        # tile spans s blocks in each direction
        for by in range(by0, by0 + s):
            for bx in range(bx0, bx0 + s):
                assert 0 <= by < num_blocks_y and 0 <= bx < num_blocks_x, f"Block index out of bounds: {(by, bx)} vs {(num_blocks_y, num_blocks_x)}"
                block_sum[by, bx] += score.item()
                block_count[by, bx] += 1.0

    # Average score per block
    block_scores = block_sum / block_count
    return block_scores
