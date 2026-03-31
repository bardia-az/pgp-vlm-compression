from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import io
import os
import cv2
import json
from functools import partial
from filelock import FileLock
import subprocess
import numpy as np
from compressai.zoo import image_models as codec_zoo
from compressai.ops import compute_padding
from torchvision import transforms
import torch.nn.functional as F


register_heif_opener()


def compress_image(img: Image.Image, quality: int, format: str = "JPEG") -> tuple[Image, float]:
    if format == "VVC":
        return compress_image_vvc(img, qp=quality)
    elif format == "LIC":
        return compress_image_compressai(img, quality=quality)
    else:
        return compress_image_pil(img, quality=quality, format=format)

def compress_image_compressai(img: Image.Image, quality: int = 1) -> tuple[Image.Image, float]:
    model = codec_zoo["cheng2020-anchor"](quality=quality, metric="mse", pretrained=True)
    model.eval()
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    h, w = img_tensor.size(2), img_tensor.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    im_padded = F.pad(img_tensor, pad, mode="constant", value=0)
    out = model.compress(im_padded)
    x_hat = model.decompress(**out)["x_hat"]
    x_hat = F.pad(x_hat, unpad)
    bpp = sum(len(s[0]) for s in out["strings"]) * 8.0 / (img.width * img.height)
    compressed_img = transforms.ToPILImage()(x_hat.squeeze(0))
    return compressed_img, bpp

def compress_image_pil(img: Image.Image, quality, format="JPEG") -> tuple[Image.Image, float]:
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=quality)
    size_bytes = buffered.tell()
    bpp = (size_bytes * 8) / (img.width * img.height)
    buffered.seek(0)
    compressed_img = Image.open(buffered)
    return compressed_img, bpp

# Pre-allocate temp files in RAM
YUV_TMP_FILE = "/dev/shm/tmp_image.yuv"
VVC_TMP_FILE = "/dev/shm/tmp_image.vvc"

def pil_to_yuv420(img: Image.Image) -> np.ndarray:
    """Convert PIL image to YUV420 planar format (8-bit)."""
    img = img.convert("RGB")
    rgb = np.array(img, dtype=np.uint8)

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    y = ((0.257*r + 0.504*g + 0.098*b) + 16).astype(np.uint8)
    u = ((-0.148*r - 0.291*g + 0.439*b) + 128).astype(np.uint8)
    v = ((0.439*r - 0.368*g - 0.071*b) + 128).astype(np.uint8)

    u_420 = u[::2, ::2]
    v_420 = v[::2, ::2]

    yuv = np.concatenate([y.flatten(), u_420.flatten(), v_420.flatten()])
    return yuv

def yuv420_to_pil(yuv: np.ndarray, width: int, height: int) -> Image.Image:
    """Convert YUV420 planar (8-bit) numpy array to PIL RGB image."""
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    y = yuv[:y_size].reshape((height, width)).astype(np.float32)
    u = yuv[y_size:y_size + uv_size].reshape((height // 2, width // 2)).astype(np.float32)
    v = yuv[y_size + uv_size:].reshape((height // 2, width // 2)).astype(np.float32)

    # Upsample chroma planes to full resolution
    u_up = u.repeat(2, axis=0).repeat(2, axis=1)
    v_up = v.repeat(2, axis=0).repeat(2, axis=1)

    # Standard BT.601 limited-range conversion
    y -= 16
    u_up -= 128
    v_up -= 128

    r = np.clip(1.164 * y + 1.596 * v_up, 0, 255)
    g = np.clip(1.164 * y - 0.392 * u_up - 0.813 * v_up, 0, 255)
    b = np.clip(1.164 * y + 2.017 * u_up, 0, 255)

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

def compress_with_vvenc(yuv_file: str, width: int, height: int, qp: int = 32, bitstream_file: str = VVC_TMP_FILE):
    cmd = [
        "vvencapp",
        "--input", yuv_file,
        "--size", f"{width}x{height}",
        "--format", "yuv420",
        "--preset", "medium",
        "--fps", "1",
        "--qpa", "0",
        "--qp", str(qp),
        "--internal-bitdepth", "8",
        "--output", bitstream_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return bitstream_file

def decode_with_vvdec(bitstream_file: str, width: int, height: int) -> np.ndarray:
    yuv_out = bitstream_file.replace(".vvc", "_dec.yuv")
    cmd = [
        "vvdecapp",
        "--bitstream", bitstream_file,
        "--frames", "1",
        "--output", yuv_out,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    with open(yuv_out, "rb") as f:
        yuv = np.frombuffer(f.read(), dtype=np.uint8)
    # os.remove(yuv_out)
    return yuv

def get_bpp(bitstream_file: str, width: int, height: int) -> float:
    size_bytes = os.path.getsize(bitstream_file)
    return (size_bytes * 8) / (width * height)

def compress_image_vvc(img: Image.Image, qp: int = 32):
    """Compress a PIL image with VVenc and return decoded PIL + bpp, reusing temp files in RAM."""
    width, height = img.width, img.height
    # compute padding needed to make dimensions even for YUV420
    pad_right = width % 2
    pad_bottom = height % 2
    if pad_right or pad_bottom:
        img = ImageOps.expand(img, border=(0, 0, pad_right, pad_bottom), fill=(128, 128, 128))
        width, height = img.width, img.height
    
    yuv = pil_to_yuv420(img)

    # Overwrite same temp YUV file
    with open(YUV_TMP_FILE, "wb") as f:
        f.write(yuv.tobytes())

    # Encode
    bitstream_file = compress_with_vvenc(YUV_TMP_FILE, width, height, qp, VVC_TMP_FILE)
    bpp = get_bpp(bitstream_file, width, height)

    # Decode
    decoded_yuv = decode_with_vvdec(bitstream_file, width, height)
    decoded_img = yuv420_to_pil(decoded_yuv, width, height)

    return decoded_img, bpp


def compress_image_jpeg_cv(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_img = cv2.imencode('.jpg', img, encode_param)
    assert success, "Image encoding failed"
    bpp = encoded_img.size * 8 / (img.shape[0] * img.shape[1])
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    # Convert to PIL
    cv_img_rgb = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    compressed_image = Image.fromarray(cv_img_rgb)
    return compressed_image, bpp


def update_json_file(filepath, new_data):
    """
     Safely update a JSON file with new data (hierarchical merge),
    protecting it with a file lock so multiple processes can't
    write at the same time.
    If the file doesn't exist, it creates it.

    Args:
        filepath (str): Path to the JSON file.
        new_data (dict): New dictionary to merge into the JSON.
    """

    lock_path = filepath + ".lock"
    lock = FileLock(lock_path)

    with lock:
        # Load existing data if file exists
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Deep update helper
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)  # recurse for nested dicts
                else:
                    d[k] = v
            return d

        # Merge new data
        data = deep_update(data, new_data)

        # Save back to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    return data


def add_kwargs_to_factory(factory, **extra):
    """
    Return a new callable that behaves like `factory`
    but always receives the extra kwargs.
    Works for partials or plain callables.
    """
    if isinstance(factory, partial):
        # clone its existing args/keywords
        return partial(factory.func, *factory.args,
                       **{**factory.keywords, **extra})
    else:
        return partial(factory, **extra)