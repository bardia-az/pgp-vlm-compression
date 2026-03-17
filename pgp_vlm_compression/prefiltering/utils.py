import numpy as np
import cv2
import torch


def gaussian_blur(image: np.ndarray, sigma: float, ksize: int) -> np.ndarray:
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    return blurred


def gaussian_blur_blockwise(img: np.ndarray, block_sigma: np.ndarray, stride: int, ksize: int = 11):
    """
    Apply Gaussian blur to each stride x stride block of input image based on the block sigma values.
    """
    H, W, C = img.shape
    num_blocks_y, num_blocks_x = block_sigma.shape
    assert ksize % 2 == 1, "Kernel size must be odd"

    blurred_img = np.zeros_like(img)

    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y0, x0 = by * stride, bx * stride
            y1, x1 = y0 + stride, x0 + stride
            assert y1 <= H and x1 <= W, "Block exceeds image dimensions"
            block = img[y0:y1, x0:x1]
            sigma = block_sigma[by, bx]

            if sigma <= 1e-6:
                blurred_block = block
            else:
                blurred_block = cv2.GaussianBlur(block, (ksize, ksize), sigmaX=sigma)

            blurred_img[y0:y1, x0:x1] = blurred_block

    return blurred_img


def get_sigma_values(scores: torch.Tensor, sigma_min: float = 0.5, sigma_max: float = 5.0, method: str = "linear") -> torch.Tensor:
    # Compute sigma values based on the scores
    if method == "linear":
        return sigma_max - (sigma_max - sigma_min) * scores
    elif method == "exponential":
        return sigma_min * (sigma_max / sigma_min) ** (1 - scores)
    elif method == "inverse":
        return (sigma_max * sigma_min) / (scores * (sigma_max - sigma_min) + sigma_min)
    else:
        raise ValueError(f"Unknown method: {method}")
    

def get_adaptive_stride(H, W, tile_size=224, target_tiles=24):
    """
    Compute an adaptive stride that divides the tile_size exactly.

    Args:
        H, W: image height and width
        tile_size: size of the CLIP tile
        target_tiles: desired approximate number of tiles

    Returns:
        stride: one of tile_size, tile_size//2, tile_size//4
    """
    # Candidate strides that divide the tile size exactly
    candidate_strides = [tile_size // 4, tile_size // 2, tile_size]

    # Compute approximate number of tiles for each candidate
    best_stride = candidate_strides[0]
    min_diff = float('inf')

    for stride in candidate_strides:
        num_tiles_x = max(1, int(np.ceil((W - tile_size) / stride)) + 1)
        num_tiles_y = max(1, int(np.ceil((H - tile_size) / stride)) + 1)
        num_tiles = num_tiles_x * num_tiles_y

        diff = abs(num_tiles - target_tiles)
        if diff <= min_diff:
            min_diff = diff
            best_stride = stride

    return best_stride
