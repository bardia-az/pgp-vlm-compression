import numpy as np
import cv2
from pathlib import Path
import os
import datetime
import torch
import time

from pgp_vlm_compression.prefiltering.tiling import *
from pgp_vlm_compression.prefiltering.utils import *
from pgp_vlm_compression.prefiltering.text import preproc_prompt_clip


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# Timing log: set PREFILTER_TIMINGS_FILE to a path (e.g. "prefilter_timings.csv") to record each CLIP-forward time for later stats.
def _log_timing(elapsed_ms: float, target_num_tiles: int, width: int, height: int) -> None:
    path = os.environ.get("PREFILTER_TIMINGS_FILE", "")
    if not path:
        return
    p = Path(path)
    write_header = not p.exists() or p.stat().st_size == 0
    with open(path, "a") as f:
        if write_header:
            f.write("elapsed_ms,target_num_tiles,width,height,timestamp\n")
        f.write(f"{elapsed_ms:.4f},{target_num_tiles},{width},{height},{datetime.datetime.utcnow().isoformat()}Z\n")


def prefilter_image(
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    image: Image.Image,
    prompt: str,
    logit_scale: float = 100.0,
    sigma_min: float = 0.5,
    sigma_max: float = 5.0,
    method: str = "exponential",
    ksize: int = 5,
    strict_text_summ: bool = False,
    target_num_tiles: int = 24,
    preserve_size: bool = True,
    pre_text_summ: bool = True,
    preproc_prompt: bool = True,
) -> Image.Image:

    W, H = image.size
    tile_size = clip_model.visual.image_size[0]
    stride = get_adaptive_stride(H, W, tile_size=tile_size, target_tiles=target_num_tiles)

    pixel_values_clip, positions, padded_img, scale, pad = preprocess_image_sliding_window(
        image,
        clip_preprocess,
        input_size=tile_size,
        stride=stride
    )

    pixel_values_clip = pixel_values_clip.cuda()
    if preproc_prompt:
        prompt = preproc_prompt_clip(prompt, strict_mode=strict_text_summ, pre_text_summ=pre_text_summ)

    
    torch.cuda.synchronize()
    start = time.perf_counter()

    ids_clip = clip_tokenizer([prompt]).cuda()

    with torch.no_grad():
        clip_out = clip_model.encode_image(pixel_values_clip)
        image_features = clip_out if isinstance(clip_out, torch.Tensor) else clip_out[0]
        text_features = clip_model.encode_text(ids_clip)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        orig_scores = image_features_norm @ text_features_norm.T
        tile_scores = torch.softmax(logit_scale * orig_scores, dim=0) * orig_scores.shape[0]

    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"Time taken: {elapsed_ms:.2f} ms")
    _log_timing(elapsed_ms, target_num_tiles, W, H)

    score_map = merge_score_tiles(tile_scores, positions, img_size=padded_img.size, tile_size=tile_size, stride=stride)

    sigma_map = get_sigma_values(score_map, sigma_min=sigma_min, sigma_max=sigma_max, method=method)

    cv_img = np.array(padded_img)           # RGB (from PIL)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # convert to BGR
    filtered_padded = gaussian_blur_blockwise(cv_img, sigma_map.cpu().numpy(), stride=stride, ksize=ksize)

    # Crop to original size and resize back to original resolution
    if preserve_size:
        left, top, right, bottom = pad
        cropped = filtered_padded[top : filtered_padded.shape[0] - bottom, left : filtered_padded.shape[1] - right]
        final_img = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        final_img = filtered_padded

    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(final_img)

    return pil_img
