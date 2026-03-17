import numpy as np
import cv2
from pathlib import Path
import argparse
import os
import datetime
from matplotlib import pyplot as plt
import torch
import time

from pgp_vlm_compression import TinyCLIP
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

def prefilter_image_old(
    clip_model,
    clip_tokenizer,
    clip_preprocess,
    image: Image,
    prompt: str,
    max_tiles: int = 12,
    logit_scale: float = 100.0,
    sigma_min: float = 0.5,
    sigma_max: float = 5.0,
    method: str = "exponential",
    ksize: int = 5,
    strict_text_summ: bool = False
) -> np.ndarray:
    
    tiles, pixel_values_clip, grid_size = load_image_for_prefiltering(
        image,
        clip_preprocess,
        input_size=clip_model.visual.image_size[0],
        max_num=max_tiles,
    )
    pixel_values_clip = pixel_values_clip.cuda()
    # Preprocess prompt
    prompt = preproc_prompt_clip(prompt, strict_mode=strict_text_summ)
    ids_clip = clip_tokenizer([prompt])
    ids_clip = ids_clip.cuda()
    # Run the CLIP model
    with torch.no_grad():
        clip_out = clip_model.encode_image(pixel_values_clip)
        image_features = clip_out if isinstance(clip_out, torch.Tensor) else clip_out[0]
        text_features = clip_model.encode_text(ids_clip)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        orig_scores = image_features_norm @ text_features_norm.T
        orig_scores = torch.softmax(logit_scale * orig_scores, dim=0) * orig_scores.shape[0]

    # Apply Gaussian blurring to the image
    sigma_values = get_sigma_values(orig_scores, sigma_min=sigma_min, sigma_max=sigma_max, method=method)
    blurred_images_list = []
    for tile, sigma in zip(tiles, sigma_values):
        cv_img = np.array(tile)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        blurred_image = gaussian_blur(cv_img, sigma=float(sigma.item()), ksize=ksize)
        blurred_images_list.append(blurred_image)
    
    batched_images = np.stack(blurred_images_list, axis=0)  # shape: (N, H, W, C)
    merged_image = merge_patches_np(batched_images, grid_size)
    return merged_image


def tinyclip_filter(
    clip_arch: str = "TinyCLIP-ViT-39M-16-Text-19M",
    clip_pretrained: str = "YFCC15M",
    prompt: str = "Describe the image shortly",
    img_path: str = ROOT / "data/sample_imgs/tennis.jpg",
    use_thumbnail: bool = False,
    max_tiles: int = 12,
    prescale: float = 100.0,
    sigma_min: float = 0.5,
    sigma_max: float = 5.0,
    method: str = "exponential",
    ksize: int = 5,
) -> np.ndarray:
    # Setup CLIP model
    clip, _, clip_preprocess = TinyCLIP.create_model_and_transforms(clip_arch, pretrained=clip_pretrained)
    clip = clip.cuda()
    clip_tokenizer = TinyCLIP.get_tokenizer(clip_arch)

    # Data preparation
    tiles, pixel_values_clip, grid_size = load_image_for_prefiltering(
        img_path,
        clip_preprocess,
        max_num=max_tiles,
    )
    pixel_values_clip = pixel_values_clip.cuda()
    ids_clip = clip_tokenizer([prompt])

    # Run the original CLIP model
    image_features, _ = clip.encode_image(pixel_values_clip)
    text_features = clip.encode_text(ids_clip)
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    orig_scores = image_features_norm @ text_features_norm.T
    orig_scores = torch.softmax(prescale * orig_scores, dim=0) * orig_scores.shape[0]
    sigma_values = get_sigma_values(orig_scores, sigma_min=sigma_min, sigma_max=sigma_max, method=method)

    # Apply Gaussian blur to the image
    blurred_images_list = []
    for tile, sigma in zip(tiles, sigma_values):
        cv_img = np.array(tile)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        blurred_image = gaussian_blur(cv_img, sigma=float(sigma.item()), ksize=ksize)
        blurred_images_list.append(blurred_image)
    
    batched_images = np.stack(blurred_images_list, axis=0)  # shape: (N, H, W, C)
    # patches = combine_quadrants_into_patches_np(batched_images)
    merged_image = merge_patches_np(batched_images, grid_size)
    return merged_image

if __name__ == "__main__":
    ### Testing the Gaussian Blur function
    # parser = argparse.ArgumentParser(description="Apply Gaussian blur to an image.")
    # parser.add_argument("--input_image", type=str, default=ROOT / "data/sample_imgs/tennis.jpg", help="Path to the input image.")
    # parser.add_argument("--output_image", type=str, default=ROOT / "blurred.jpg", help="Path to save the blurred image.")
    # parser.add_argument("--sigma", type=float, default=10.0, help="Sigma value for Gaussian blur.")
    # parser.add_argument("--ksize", type=int, default=5, help="Kernel size for Gaussian blur.")
    # args = parser.parse_args()
    # test_image = cv2.imread(str(args.input_image))
    # filtered_image = gaussian_blur(test_image, args.sigma, args.ksize)
    # cv2.imwrite(str(args.output_image), filtered_image)

    ### Testing the Tiny CLIP filter function
    parser = argparse.ArgumentParser(description="Apply Gaussian blur to an image based on prompt.")
    parser.add_argument("--input_image", type=str, default=ROOT / "data/sample_imgs/tennis.jpg", help="Path to the input image.")
    parser.add_argument("--output_image", type=str, default=ROOT / "blurred_clip.jpg", help="Path to save the blurred image.")
    parser.add_argument("--sigma_min", type=float, default=0.5, help="Minimum sigma value for Gaussian blur.")
    parser.add_argument("--sigma_max", type=float, default=10.0, help="Maximum sigma value for Gaussian blur.")
    parser.add_argument("--method", type=str, default="exponential", choices=["linear", "exponential", "inverse"], help="Method for sigma value calculation.")
    parser.add_argument("--ksize", type=int, default=11, help="Kernel size for Gaussian blur.")
    parser.add_argument("--prescale", type=float, default=100.0, help="Prescale factor for the scores before softmax.")
    parser.add_argument("--clip_arch", type=str, default="TinyCLIP-ViT-39M-16-Text-19M", help="Architecture of the CLIP model.")
    parser.add_argument("--clip_pretrained", type=str, default="YFCC15M", help="Pretrained weights for the CLIP model.")
    parser.add_argument("--prompt", type=str, default="Describe the image shortly", help="Prompt for the CLIP model.")
    parser.add_argument("--use_thumbnail", action='store_true', help="Use thumbnail for the input image.")
    args = parser.parse_args()

    prefiltered_img = tinyclip_filter(
        img_path=args.input_image,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        method=args.method,
        ksize=args.ksize,
        prescale=args.prescale,
        clip_arch=args.clip_arch,
        clip_pretrained=args.clip_pretrained,
        prompt=args.prompt,
        use_thumbnail=args.use_thumbnail
    )
    cv2.imwrite(str(args.output_image), prefiltered_img)

    ### Testing the sigma value generation function
    # draw the returned values of get_sigma_values for all the possible scores between 0 and 1 for all the possible methods on a same plot
    # scores = torch.linspace(0, 15, steps=100)
    # methods = ["linear", "exponential", "inverse"]
    # plt.figure(figsize=(10, 6))
    # for method in methods:
    #     sigmas = get_sigma_values(scores, method=method)
    #     plt.plot(scores.numpy(), sigmas.numpy(), label=method)
    # plt.xlabel("Scores")
    # plt.ylabel("Sigma Values")
    # plt.title("Sigma Values for Different Methods")
    # plt.legend()
    # plt.grid()
    # plt.show()
