from PIL import Image
from functools import partial
import torch
import argparse
import os
import open_clip

from vlmeval.vlm.llava import LLaVA_OneVision_HF
from vlmeval.config import supported_VLM

from pgp_vlm_compression.utils.utils import compress_image
from pgp_vlm_compression.prefiltering.prefilter import prefilter_image
from pgp_vlm_compression import TinyCLIP


def load_image_preproc(
    image_file: str,
    prompt: str,
    args: argparse.Namespace
):
    orig_image = Image.open(image_file).convert('RGB')
    if args.show_image:
        orig_image.show()
    file_size_bytes = os.path.getsize(image_file)  # on-disk size
    width, height = orig_image.size
    num_pixels = width * height
    orig_bpp = (file_size_bytes * 8) / num_pixels
    compressed_bpp = 0

    if args.prefilter:
        image_filtered = prefilter_image(
            clip_model=args.clip_model,
            clip_tokenizer=args.clip_tokenizer,
            clip_preprocess=args.clip_preprocess,
            image=orig_image,
            prompt=prompt,
            target_num_tiles=args.clip_tiles_num,
            logit_scale=args.logit_scale,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            method=args.method,
            ksize=args.ksize,
            strict_text_summ=args.strict_text_summ,
            preserve_size=args.preserve_size,
            pre_text_summ=args.pre_text_summ,
            preproc_prompt=args.preproc_prompt
        )
        image, compressed_bpp = compress_image(image_filtered, quality=args.compress_ratio, format=args.compress_format)
        if args.show_image:
            image.show()
    else:
        image = orig_image
        if args.compress_ratio != 0:
            image, compressed_bpp = compress_image(image, quality=args.compress_ratio, format=args.compress_format)
            if args.show_image:
                image.show()

    return orig_image, image, orig_bpp, compressed_bpp


class LLaVAWithPreproc(LLaVA_OneVision_HF):
    """ LLaVA model with image pre-processing (compression and pre-filtering) 
    Args:
        preproc_args: arguments for pre-processing (namespace)
        kwargs: arguments for the original LLaVA_OneVision_HF (dict)

    Note: Preprocessing is only supported for image datasets and not for video datasets.
    """
    def __init__(self, preproc_args, **kwargs):
        super().__init__(**kwargs)
        # Setup CLIP model for prefiltering
        clip_lib = TinyCLIP if preproc_args.clip_arch.startswith('TinyCLIP') else open_clip
        clip_model, _, clip_preprocess = clip_lib.create_model_and_transforms(preproc_args.clip_arch, pretrained=preproc_args.clip_pretrained)
        clip_model = clip_model.cuda()
        clip_tokenizer = clip_lib.get_tokenizer(preproc_args.clip_arch)
        preproc_args.clip_model = clip_model
        preproc_args.clip_tokenizer = clip_tokenizer
        preproc_args.clip_preprocess = clip_preprocess
        self.preproc_args = preproc_args

    def generate_inner_image(self, message, dataset=None):
        content, images, orig_images = "", [], []
        image_sizes = []

        clip_prompt = ' '.join([x['value'] for x in message if x['type'] == 'text'])
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                orig_img, img, orig_bpp, compressed_bpp = load_image_preproc(
                    msg["value"], clip_prompt, self.preproc_args)
                images.append(img)
                orig_images.append(orig_img)
                image_sizes.append(img.size)
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=2048)
        response = self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        if self.preproc_args.show_image:
            inputs = self.processor(images=orig_images, text=prompt, return_tensors="pt").to('cuda', torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=2048)
            orig_response = self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Response with preprocessed image: {response}")
            print(f"Response with original image: {orig_response}")

        return response, orig_bpp, compressed_bpp
    

llava_series = {
    "llava-onevision-qwen2-0.5b-ov-hf": partial(
        LLaVAWithPreproc, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    ),
    "llava-onevision-qwen2-0.5b-si-hf": partial(
        LLaVAWithPreproc, model_path="llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    ),
    "llava-onevision-qwen2-7b-ov-hf": partial(
        LLaVAWithPreproc, model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    ),
    "llava-onevision-qwen2-7b-si-hf": partial(
        LLaVAWithPreproc, model_path="llava-hf/llava-onevision-qwen2-7b-si-hf"
    ),
}

supported_VLM.update(llava_series)