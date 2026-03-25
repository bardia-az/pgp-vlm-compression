from PIL import Image
from functools import partial
import torch
import argparse
import os
import open_clip

from vlmeval.vlm.internvl.internvl_chat import *
from vlmeval.vlm.internvl.utils import *
from vlmeval.config import supported_VLM

from pgp_vlm_compression.utils.utils import compress_image
from pgp_vlm_compression.prefiltering.prefilter import prefilter_image
from pgp_vlm_compression import TinyCLIP


def load_image_preproc(
    image_file: str,
    prompt: str,
    args: argparse.Namespace,
    input_size: int = 448,
    max_num=6,
    upscale=False,
):
    orig_image = Image.open(image_file).convert('RGB')
    if upscale:
        orig_image = orig_image.resize((orig_image.width * 2, orig_image.height * 2), Image.BILINEAR)
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
            preproc_prompt=args.preproc_prompt,
        )
        image, compressed_bpp = compress_image(
            image_filtered,
            quality=args.compress_ratio,
            format=args.compress_format,
        )
        if args.show_image:
            image.show()
    else:
        image = orig_image
        if args.compress_ratio != 0:
            image, compressed_bpp = compress_image(image, quality=args.compress_ratio, format=args.compress_format)
            if args.show_image:
                image.show()

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    if args.show_image:
        orig_images = dynamic_preprocess(orig_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        orig_pixel_values = [transform(image) for image in orig_images]
        orig_pixel_values = torch.stack(orig_pixel_values)
    else:
        orig_pixel_values = None
    return pixel_values, orig_pixel_values, orig_bpp, compressed_bpp


class InternVLWithPreproc(InternVLChat):
    """ InternVL model with image pre-processing (compression and pre-filtering) 
    Args:
        preproc_args: arguments for pre-processing (namespace)
        kwargs: arguments for the original InternVLChat (dict)

    Note: Preprocessing is only supported for the V2.0 models (InternVL2 onwards).
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

    @torch.no_grad()
    def generate_v2(self, message, dataset=None):

        use_mpo_prompt = self.use_mpo_prompt and (self.use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        clip_prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])

        if dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO':
            prompt = build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list, orig_pixel_values_list = [], [], []
            orig_bpp, compressed_bpp = 0, 0
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values, curr_orig_pixel_values, curr_orig_bpp, curr_compressed_bpp = load_image_preproc(
                    file_name, max_num=max_num, upscale=upscale_flag, args=self.preproc_args, prompt=clip_prompt)
                curr_pixel_values = curr_pixel_values.to(self.device).to(torch.bfloat16)
                if self.preproc_args.show_image:
                    curr_orig_pixel_values = curr_orig_pixel_values.to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
                orig_pixel_values_list.append(curr_orig_pixel_values)
                orig_bpp += curr_orig_bpp
                compressed_bpp += curr_compressed_bpp
            pixel_values = torch.cat(pixel_values_list, dim=0)
            orig_pixel_values = torch.cat(orig_pixel_values_list, dim=0) if self.preproc_args.show_image else None
            orig_bpp /= image_num
            compressed_bpp /= image_num
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values, orig_pixel_values, orig_bpp, compressed_bpp = load_image_preproc(
                image_path, max_num=max_num, upscale=upscale_flag, args=self.preproc_args, prompt=clip_prompt)
            pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
            if self.preproc_args.show_image:
                orig_pixel_values = orig_pixel_values.to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values, orig_pixel_values, orig_bpp, compressed_bpp = None, None, 0, 0
            num_patches_list = []

        response_list = []
        for idx in range(self.best_of_n):
            kwargs_default = self.kwargs.copy()
            kwargs_default['do_sample'] = idx > 0
            kwargs_default['temperature'] = 0.7
            kwargs_default['top_p'] = 0.95

            if self.use_lmdeploy:       # Doesn't support preproc_args.show_image
                from lmdeploy import GenerationConfig
                gen_config = GenerationConfig(**kwargs_default)
                gen_config.random_seed = None
                messages_list = prepare_messages_list(prompt, image_path, system_prompt=self.system_prompt)
                assert len(messages_list) == 1
                response = self.model(messages_list, gen_config=gen_config)[0]
                response = response.text
            else:
                if self.system_prompt is not None:
                    self.model.system_message = self.system_prompt
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=prompt,
                    generation_config=kwargs_default,
                    verbose=idx == 0,
                )
                if self.preproc_args.show_image and orig_pixel_values is not None:
                    response_orig = self.model.chat(
                        self.tokenizer,
                        pixel_values=orig_pixel_values,
                        num_patches_list=num_patches_list,
                        question=prompt,
                        generation_config=kwargs_default,
                        verbose=idx == 0,
                    )
                    print(f"Response with preprocessed image: {response}")
                    print(f"Response with original image: {response_orig}")
            response_list.append(response)

        if self.best_of_n > 1:
            response_list = self.reward_model.select_best_response(
                tokenizer=self.reward_tokenizer,
                question=prompt,
                response_list=response_list,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
            )
        response = response_list[0]

        if dataset is not None and not listinstr(['WeMath'], dataset):
            if use_mpo_prompt:
                response = mpo_post_processing(response, dataset)
            elif self.use_cot and self.use_postprocess:
                response = extract_boxed_content(response)

        if dataset is not None and DATASET_TYPE(dataset) == 'GUI' and self.screen_parse:
            # Parse the bounding box coordinates from the response
            response = parse_bbox_internvl(response)
            # Normalize the coordinates to the range [0, 1]
            if isinstance(response, list):
                response = [item / 1000 for item in response]
                # Convert the coordinates to the format required by the GUI
                response = f"x={response[0]}, y={response[1]}"

        return response, orig_bpp, compressed_bpp


internvl2 = {
    "InternVL2-1B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-1B", version="V2.0"
    ),
    "InternVL2-2B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-2B", version="V2.0"
    ),
    "InternVL2-4B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-4B", version="V2.0"
    ),
    "InternVL2-8B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-8B", version="V2.0"
    ),
    "InternVL2-26B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-26B", version="V2.0"
    ),
    "InternVL2-40B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-40B", version="V2.0"
    ),
    "InternVL2-76B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-Llama3-76B", version="V2.0"
    ),
    "InternVL2-8B-MPO": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2-8B-MPO", version="V2.0"
    ),
    "InternVL2-8B-MPO-CoT": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
}

internvl2_5 = {
    "InternVL2_5-1B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-1B", version="V2.0"
    ),
    "InternVL2_5-2B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-2B", version="V2.0"
    ),
    "QTuneVL1-2B": partial(
        InternVLWithPreproc, model_path="hanchaow/QTuneVL1-2B", version="V2.0"
    ),
    "InternVL2_5-4B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-4B", version="V2.0"
    ),
    "InternVL2_5-8B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0"
    ),
    "InternVL2_5-26B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-26B", version="V2.0"
    ),
    "InternVL2_5-38B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-38B", version="V2.0"
    ),
    "InternVL2_5-78B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-78B", version="V2.0"
    ),
    # InternVL2.5 series with Best-of-N evaluation
    "InternVL2_5-8B-BoN-8": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0",
        best_of_n=8, reward_model_path="OpenGVLab/VisualPRM-8B",
    ),
}

internvl2_5_mpo = {
    "InternVL2_5-1B-MPO": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2_5-1B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-2B-MPO": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2_5-2B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-4B-MPO": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2_5-4B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-MPO": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2_5-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-26B-MPO": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2_5-26B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-38B-MPO": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2_5-38B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-78B-MPO": partial(
        InternVLWithPreproc,
        model_path="OpenGVLab/InternVL2_5-78B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-GUI": partial(
        InternVLWithPreproc,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/models/internvl2_5_8b_internlm2_5_7b_dynamic_res_stage1", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
     "InternVL3-7B-GUI": partial(
        InternVLWithPreproc,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/GUI/checkpoints/internvl3_7b_dynamic_res_stage1_56/", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
}

internvl3 = {
    "InternVL3-1B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL3-1B", version="V2.0"
    ),
    "InternVL3-2B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL3-2B", version="V2.0"
    ),
    "InternVL3-8B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL3-8B", version="V2.0",
    ),
    "InternVL3-9B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL3-9B", version="V2.0"
    ),
    "InternVL3-14B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL3-14B", version="V2.0"
    ),
    "InternVL3-38B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL3-38B", version="V2.0"
    ),
    "InternVL3-78B": partial(
        InternVLWithPreproc, model_path="OpenGVLab/InternVL3-78B", version="V2.0"
    ),
}

internvl_groups = [internvl2, internvl2_5, internvl2_5_mpo, internvl3]
internvl_series = {}
for group in internvl_groups:
    internvl_series.update(group)

supported_VLM.update(internvl_series)