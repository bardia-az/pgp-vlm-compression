import bjontegaard as bd

import json
import matplotlib.pyplot as plt
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def calculate_bdrate(
    json_file,
    sets,
    labels,
    pairs,
    category,
):
    """
    Plot rate-accuracy curves for several dataset/model pairs.

    Parameters
    ----------
    json_file : str
        Path to results JSON.
    sets : list[list[str]]
        Each inner list is a group of run IDs -> one curve.
    labels : list[str]
        Legend labels for each group.
    pairs : list[tuple[str, str]]
        List of (dataset, model) pairs to plot.
    categories : list[str] | None
        Metrics to plot (e.g. ["OCR"]). If None, take all from the first pair.
    """
    
    with open(json_file) as f:
        data = json.load(f)

    # Outer loop: dataset/model pair
    for ds, mdl in pairs:
        plt.figure(figsize=(7, 5))

        pts = {}
        for run_set, label in zip(sets, labels):
            rates, scores = [], []

            for run_id in run_set:
                entry = data.get(run_id, {}).get(ds, {}).get(mdl)
                if not entry:
                    # skip missing run for this pair
                    continue
                
                rate = entry.get("compressed_bpp")
                if not isinstance(category, str):
                    score = sum(entry.get(cat, 0) for cat in category)
                else:
                    score = entry.get(category)

                if rate is not None and score is not None:
                    rates.append(rate)
                    scores.append(score)
                else:
                    raise ValueError(f"Missing data for run {run_id}, dataset {ds}, model {mdl}")

            pts[label] = sorted(zip(rates, scores), key=lambda x: x[0])
            r, s = zip(*pts[label])

            plt.plot(r, s, marker="o", label=label)

        plt.title(f"{category} — {ds} / {mdl}")
        plt.xlabel("Rate (bpp)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.xlim(0, 0.8)

        anchor_rate, anchor_score = zip(*pts["anchor"])
        test_rate, test_score = zip(*pts["test"])
        bdrate = bd.bd_rate(anchor_rate, anchor_score, test_rate, test_score, method='pchip', require_matching_points=False)
        bd.plot_rcd(anchor_rate, anchor_score, test_rate, test_score, method='pchip', require_matching_points=False)
        bd.compare_methods(anchor_rate, anchor_score, test_rate, test_score, require_matching_points=False)
        print(f"BD-Rate for {ds} / {mdl} ({category}): {bdrate:.2f}%")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-result-path', type=str, default=str(ROOT / 'results.json'))
    args = parser.parse_args()

    sets = [
        # Original uncompressed
        ["prefilter_0-prescale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_0-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M"],

        # pure JPEG compression
        [
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_5-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_7-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_10-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_15-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
        ],

        # pure HEIF compression
        [
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_7-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_10-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_15-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
        ],

        # pure VVC compression
        [
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_50-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_48-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_45-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_42-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_35-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
        ],

        # pure Cheng compression
        [
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_1-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_2-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_3-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_4-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_5-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
        ],

        
        # Proposed 22M - JPEG
        ["prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
        ],

        
        # Proposed 22M - HEVC
        [
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
        ],

        # Proposed 22M - VVC
        [
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_48-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_45-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_42-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_35-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
        ],

        # Proposed 22M - Cheng
        [
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_1-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_2-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_3-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_4-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
        ],
    ]

    labels = ["anchor", "test"]
    # labels = ["test", "anchor"]

    pairs = [
    # ("MME", "InternVL3-9B"),
    # ("MMBench_DEV_EN_V11", "InternVL3-9B"),
    ("SEEDBench_IMG", "InternVL3-9B"),
    # ("MME", "llava-onevision-qwen2-7b-ov-hf"),
    # ("MMBench_DEV_EN_V11", "llava-onevision-qwen2-7b-ov-hf"),
    # ("SEEDBench_IMG", "llava-onevision-qwen2-7b-ov-hf"),
    ]

    category = "Overall"                        # Use this for SEEDBench and MMBench
    # category = ["perception", "reasoning"]    # Use this for MME

    # category = "Visual Reasoning"
    # category = "perception"
    # category = "reasoning"
    # category = "artwork"
    
    calculate_bdrate(args.json_result_path, sets, labels, pairs, category=category)