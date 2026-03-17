from typing import List
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import Tensor

from internvl.train.dataset import build_transform


def combine_quadrants_into_patches(quadrants: Tensor, add_thumbnail: bool) -> Tensor:
    """
    Combine quadrants into full patches.

    Args:
        quadrants: Tensor of shape (N*4, H, W, C), where each group of 4 is one patch.
                   H=W expected (square quadrants)
        add_thumbnail: whether to add an extra thumbnail at the end (optional)

    Returns:
        Tensor of shape (num_patches + 1 if add_thumbnail else num_patches, H*2, W*2, C)
    """
    assert quadrants.shape[-1] == quadrants.shape[-2], \
        f"Expected quadrants to be square, got {quadrants.shape[-1]}x{quadrants.shape[-2]}"
    q_dim = quadrants.shape[-1]
    p_dim = q_dim * 2
    quadrants = quadrants.view(-1, 4, 3, q_dim, q_dim)
    out = torch.zeros(
        quadrants.size(0) + (1 if add_thumbnail else 0), 3, p_dim, p_dim,
        dtype=quadrants.dtype, device=quadrants.device
    )
    if not add_thumbnail:
        out[:, :, 0:q_dim, 0:q_dim] = quadrants[:, 0, :, :, :]
        out[:, :, 0:q_dim, q_dim:p_dim] = quadrants[:, 1, :, :, :]
        out[:, :, q_dim:p_dim, 0:q_dim] = quadrants[:, 2, :, :, :]
        out[:, :, q_dim:p_dim, q_dim:p_dim] = quadrants[:, 3, :, :, :]
    else:
        out[:-1, :, 0:q_dim, 0:q_dim] = quadrants[:, 0, :, :, :]
        out[:-1, :, 0:q_dim, q_dim:p_dim] = quadrants[:, 1, :, :, :]
        out[:-1, :, q_dim:p_dim, 0:q_dim] = quadrants[:, 2, :, :, :]
        out[:-1, :, q_dim:p_dim, q_dim:p_dim] = quadrants[:, 3, :, :, :]
        ##################################### add the thumbnail to the batch   #FIXME
    return out


def combine_quadrants_into_patches_np(quadrants: np.ndarray) -> np.ndarray:
    """
    Combine quadrants into full patches.

    Args:
        quadrants: np.ndarray of shape (N*4, H, W, C), where each group of 4 is one patch.
                   H=W expected (square quadrants)

    Returns:
        np.ndarray of shape (num_patches, H*2, W*2, C)
    """
    num_quadrants, H, W, C = quadrants.shape
    assert H == W, f"Quadrants must be square, got {H}x{W}"
    assert num_quadrants % 4 == 0, "Number of quadrants must be divisible by 4"

    num_patches = num_quadrants // 4
    patch_dim = H * 2

    # Allocate output array
    out_shape = (num_patches, patch_dim, patch_dim, C)
    out = np.zeros(out_shape, dtype=quadrants.dtype)

    for i in range(num_patches):
        q0, q1, q2, q3 = quadrants[i*4:(i+1)*4]
        out[i, 0:H, 0:W] = q0  # top-left
        out[i, 0:H, W:patch_dim] = q1  # top-right
        out[i, H:patch_dim, 0:W] = q2  # bottom-left
        out[i, H:patch_dim, W:patch_dim] = q3  # bottom-right

    return out


def split_patches_into_quadrants(patches: List[Image.Image], use_thumbnail: bool) -> List[Image.Image]:
    assert patches[0].size[0] == patches[0].size[1], \
        f"Expected patches to be square, got {patches[0].size}"
    p_dim = patches[0].size[0]
    q_dim = p_dim // 2
    tiles = []
    for image in patches[:-1] if len(patches) > 1 and use_thumbnail else patches:
        tiles.append(image.crop((0, 0, q_dim, q_dim)))  # Top-left
        tiles.append(image.crop((q_dim, 0, p_dim, q_dim)))  # Top-right
        tiles.append(image.crop((0, q_dim, q_dim, p_dim)))  # Bottom-left
        tiles.append(image.crop((q_dim, q_dim, p_dim, p_dim)))  # Bottom-right
    return tiles


def save_scores_overlay_subpatches(image_tensor, scores, grid_size, save_path,
                                   patch_size=448, subpatch_size=224, alpha=0.5,
                                   cmap="jet", dpi=200, add_colorbar=False, caption=None):
    """
    Save an image with sub-patch-level scores (e.g., 224x224) overlayed as a heatmap.

    Args:
        image_tensor (torch.Tensor): [3, H, W], values in [0,1]
        scores (torch.Tensor): [n*4], scores for each 224x224 sub-patch
        grid_size (tuple): (cols, rows) grid of 448x448 patches
        save_path (str): output path (e.g. "overlay.png")
        patch_size (int): size of the large patches (default 448)
        subpatch_size (int): size of sub-patches (default 224)
        alpha (float): transparency of heatmap
        cmap (str): matplotlib colormap
        dpi (int): output resolution
        add_colorbar (bool): whether to draw a colorbar
    """
    cols, rows = grid_size
    factor = patch_size // subpatch_size  # 2 for 448→224

    # new grid size
    sub_cols = cols * factor
    sub_rows = rows * factor

    H, W = image_tensor.shape[1:]
    sub_h, sub_w = H // sub_rows, W // sub_cols

    # reshape scores correctly
    # scores: [rows*cols, 4] -> [rows, cols, 2, 2]
    scores_reshaped = scores.view(rows * cols, factor, factor)
    scores_grid = scores_reshaped.view(rows, cols, factor, factor).to(torch.float32)

    # reorder into full [sub_rows, sub_cols]
    scores_full = scores_grid.permute(0, 2, 1, 3).reshape(sub_rows, sub_cols).detach().cpu().numpy()

    # upsample scores to full image size
    score_map = np.kron(scores_full, np.ones((sub_h, sub_w)))

    # convert image to numpy
    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    im = ax.imshow(score_map, cmap=cmap, alpha=alpha, extent=[0, W, H, 0])
    ax.axis("off")

    if add_colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)

    if caption is not None:
        fig.text(0.5, -0.05, caption, ha="center", va="top", fontsize=12)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


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


def load_image_for_internvl(image_file, clip_preprocess, input_size=448, max_num=12, use_thumbnail=False):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    images, grid_size = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)

    assert pixel_values.shape[2]==pixel_values.shape[3]==448, f'Tile size of InternVl should be 448, but got {pixel_values.shape[2]}'
    tiles = split_patches_into_quadrants(images, use_thumbnail)
    pixel_values_clip = [clip_preprocess(tile) for tile in tiles]
    pixel_values_clip = torch.stack(pixel_values_clip)

    return pixel_values, pixel_values_clip, grid_size
