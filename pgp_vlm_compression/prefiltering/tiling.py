import numpy as np
from PIL import Image, ImageOps
import math
import cv2

import torch
from torch import Tensor


def merge_patches_tensor(patches: Tensor, grid_size: tuple[int, int], image_size: int = 448) -> Tensor:
    """
    Reconstruct the full image from patches stored as a tensor.

    Args:
        patches (torch.Tensor): shape [n, 3, image_size, image_size]
        grid_size (tuple): (cols, rows) used in dynamic_preprocess
        image_size (int): size of each patch (default=448)

    Returns:
        torch.Tensor: reconstructed image of shape [3, H, W]
    """
    cols, rows = grid_size
    assert patches.shape[0] == cols * rows, \
        f"Expected {cols*rows} patches, but got {patches.shape[0]}"

    # Reshape: [rows, cols, C, H, W]
    patches = patches.view(rows, cols, 3, image_size, image_size)

    # Permute to [C, rows, H, cols, W]
    patches = patches.permute(2, 0, 3, 1, 4)

    # Merge rows and cols → [C, rows*H, cols*W]
    full_img = patches.reshape(3, rows * image_size, cols * image_size)

    return full_img


def merge_patches_np(patches: np.ndarray, grid_size: tuple[int, int], image_size: int = 448) -> np.ndarray:
    """
    Reconstruct the full image from patches stored as a NumPy array.

    Args:
        patches (np.ndarray): shape [n, H, W, C]
        grid_size (tuple): (cols, rows) used in dynamic_preprocess
        image_size (int): size of each patch (default=448)

    Returns:
        np.ndarray: reconstructed image of shape [H_full, W_full, C]
    """
    cols, rows = grid_size
    n_patches = patches.shape[0]
    H, W = patches.shape[1:3]
    C = patches.shape[-1]  # assume last dim is channels
    assert n_patches == cols * rows, f"Expected {cols*rows} patches, got {n_patches}"

    # Reshape to [rows, cols, H, W, C]
    patches_reshaped = patches.reshape(rows, cols, H, W, C)

    # Merge along rows (axis=2) and cols (axis=3)
    # Step 1: merge along width (cols)
    rows_merged = np.concatenate([patches_reshaped[i, :, :, :, :] for i in range(rows)], axis=1)
    # Step 2: merge along height (rows)
    full_image = np.concatenate([rows_merged[i, :, :, :] for i in range(cols)], axis=1)

    # The above double concatenate can be simplified:
    # full_image = np.block([[patches_reshaped[i, j] for j in range(cols)] for i in range(rows)])
    
    return full_image


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, (target_aspect_ratio[0], target_aspect_ratio[1])


def load_image_for_prefiltering(image, clip_preprocess, input_size=224, max_num=12):
    tiles, grid_size = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    orig_tiles, _ = split_into_grid_tiles(image, grid_size)

    pixel_values_clip = [clip_preprocess(tile) for tile in tiles]
    pixel_values_clip = torch.stack(pixel_values_clip)

    return orig_tiles, pixel_values_clip, grid_size


def split_into_grid_tiles(img: Image.Image, grid_size: tuple[int, int]):
    """
    Divide a PIL image into tiles based on a grid size, padding if necessary.

    Args:
        img (PIL.Image): Input image.
        grid_size (tuple): (cols, rows) grid size.

    Returns:
        tiles (list of PIL.Image): List of tiles row by row.
        padded_img (PIL.Image): The padded image used for tiling.
    """
    cols, rows = grid_size
    w, h = img.size

    # Compute tile dimensions (ceil to ensure coverage)
    tile_w = math.ceil(w / cols)
    tile_h = math.ceil(h / rows)

    # Compute padded dimensions
    pad_w = tile_w * cols
    pad_h = tile_h * rows

    # Pad the image (on right and bottom)
    padded_img = ImageOps.pad(img, (pad_w, pad_h), method=Image.Resampling.BOX, color=(0, 0, 0), centering=(0.5, 0.5))

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tile = padded_img.crop((left, upper, right, lower))
            tiles.append(tile)

    return tiles, padded_img


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

    # # number of tiles along each dimension: >= 1
    # n_tiles_x = max(1, math.ceil((orig_w - tile_size) / stride) + 1)
    # n_tiles_y = max(1, math.ceil((orig_h - tile_size) / stride) + 1)

    # # resize the image preserving aspect ratio
    # target_w = (n_tiles_x - 1) * stride + tile_size
    # target_h = (n_tiles_y - 1) * stride + tile_size

    target_w = math.ceil(orig_w / tile_size) * tile_size
    target_h = math.ceil(orig_h / tile_size) * tile_size

    # scale_w = target_w / orig_w
    # scale_h = target_h / orig_h
    # scale = min(scale_w, scale_h)

    # resized_w = int(round(orig_w * scale))
    # resized_h = int(round(orig_h * scale))
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
    # padded_img = ImageOps.expand(image_resized, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(128, 128, 128))

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
