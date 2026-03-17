import json
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]



def plot_rate_accuracy_paper_ablation(
    json_file,
    sets,
    labels,
    pairs,
    category,
    plot_xlim=(0, 0.8),
    output_dir=Path("."),
):
    """
    Plot rate-accuracy curves optimized for publication quality.
    Styles are now semantically mapped based on labels.
    """
    
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    
    # --- 1. NEW: DEFINE SEMANTIC STYLE MAPS ---
    # Define the base methods. The order here determines color/marker.
    # We use simple keys that will be checked against the labels.
    
    # Create style mappings
    # colors = sns.color_palette("colorblind", n_colors=len(labels)-1) # Exclude one for "Original"
    colors = [
    "#D55E00",  # 4. Vermillion (Red-Orange)
    "#332288",  # 7. Indigo (Dark Blue/Purple) - Good for high contrast
    "#0072B2",  # 5. Blue
    "#56B4E9",  # 2. Sky Blue
    "#009E73",  # 3. Bluish Green
    "#E69F00",  # 1. Orange
    "#CC79A7",  # 6. Reddish Purple
]
    line_styles = ["-", "-", "-", "-", "-", "-", "-"]
    # markers = ["o", "s", "^"] # One marker per base method
    
    # Maps base method to a color or marker
    # color_map = {method: color for method, color in zip(base_methods, colors)}
    # marker_map = {method: marker for method, marker in zip(base_methods, markers)}
    # --- END OF NEW SECTION ---

    with open(json_file) as f:
        data = json.load(f)

    for ds, mdl in pairs:
        fig, ax = plt.subplots(figsize=(5.5, 4)) # Slightly adjusted size
        all_scores = []

        # We no longer need 'color_idx'
        for i, (run_set, label) in enumerate(zip(sets, labels)):
            rates, scores = [], []

            for run_id in run_set:
                entry = data.get(run_id, {}).get(ds, {}).get(mdl)
                if not entry:
                    continue
                
                rate = entry.get("orig_bpp") if label.lower() == "original" else entry.get("compressed_bpp")
                
                if not isinstance(category, str):
                    score = sum(entry.get(cat, 0) for cat in category)
                else:
                    score = entry.get(category)

                if rate is not None and score is not None:
                    rates.append(rate)
                    scores.append(score)

            if rates and scores:
                pts = sorted(zip(rates, scores), key=lambda x: x[0])
                r, s = zip(*pts)
                all_scores.extend(s)

                if label.lower() == "original" and len(r) == 1:
                    # This logic for "Original" remains the same
                    ax.hlines(
                        y=s[0],
                        xmin=plot_xlim[0],
                        xmax=plot_xlim[1],
                        linestyles="dotted",
                        label=f"{label}",
                        color="black",
                        linewidth=2,
                    )
                else:
                    
                    # Get styles from maps, with a default
                    current_color = colors[i - 1] if i > 0 else "black" # Original is black
                    current_marker = "o"
                    
                    # Set linestyle based on your rule
                    # current_linestyle = "--" if "Proposed" in label else "-"
                    current_linestyle = line_styles[i-1] if i > 0 else "."
                    
                    ax.plot(
                        r, 
                        s, 
                        marker=current_marker,       # From map
                        linestyle=current_linestyle, # From logic
                        label=label,
                        color=current_color,         # From map
                        linewidth=2,
                        markersize=6,
                    )
                    # --- END OF MODIFIED SECTION ---

        # --- Finalize Plot Aesthetics ---
        category_name = category if isinstance(category, str) else "Combined Accuracy"
        
        mdl_name = "InternVL3-9B" if "internvl" in mdl.lower() else "LLaVA-OV-7B" if "llava" in mdl.lower() else mdl
        ds_name = "MMBench" if "mmbench" in ds.lower() else "SEEDBench" if "seedbench" in ds.lower() else ds
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize="large")
        ax.set_xlabel("Rate (bpp)", fontsize="medium")
        ax.set_ylabel("Accuracy", fontsize="medium")
        ax.set_xlim(plot_xlim)
        
        if all_scores:
            min_s, max_s = min(all_scores), max(all_scores)
            margin = (max_s - min_s) * 0.05
            # Add a bit of extra top margin for the legend
            ax.set_ylim(min_s - margin, max_s + (margin * 1.5)) 

        # Place legend nicely
        ax.legend(frameon=False, fontsize="small", loc="lower right", handlelength=2.5)
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        sns.despine(ax=ax)
        
        fig.tight_layout()

        save_name = f"abb_rate_accuracy_rebuttal_{ds_name}_{mdl_name}.pdf"
        fig.savefig(
            output_dir / save_name, 
            bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved: {output_dir / save_name}")


def plot_rate_accuracy_paper(
    json_file,
    sets,
    labels,
    pairs,
    category,
    plot_xlim=(0, 0.8),
    output_dir=Path("."),
):
    """
    Plot rate-accuracy curves optimized for publication quality.
    Styles are now semantically mapped based on labels.
    """
    
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    
    # --- 1. NEW: DEFINE SEMANTIC STYLE MAPS ---
    # Define the base methods. The order here determines color/marker.
    # We use simple keys that will be checked against the labels.
    base_methods = ["JPEG", "HEVC", "VVC", "Cheng"]
    
    # Create style mappings
    colors = sns.color_palette("colorblind", n_colors=len(base_methods))
    markers = ["o", "s", "^", "x"] # One marker per base method

    # blue_hex = "#0072B2"  # A distinct blue (often the 5th color in the standard palette)
    # orange_hex = "#D55E00" # A vermillion red/orange (often the 4th color)
    # green_hex = "#009E73" # A distinct greenish-blue (often the 3rd color)
    # black_hex = "#000000" # The problematic 4th/7th color

    # # Instead of using the black one, we skip it and use the next good color, e.g., a purple
    # purple_hex = "#CC79A7" # Reddish Purple (often the 6th color)

    # # 2. Manually construct the final list of 4 colors in the desired order:
    # final_colors = [
    #     blue_hex,       # 1st Method
    #     orange_hex,     # 2nd Method
    #     green_hex,      # 3rd Method
    #     purple_hex      # 4th Method (Skipped black, used purple instead)
    # ]
    
    # Maps base method to a color or marker
    color_map = {method: color for method, color in zip(base_methods, colors)}
    marker_map = {method: marker for method, marker in zip(base_methods, markers)}
    # --- END OF NEW SECTION ---

    with open(json_file) as f:
        data = json.load(f)

    for ds, mdl in pairs:
        fig, ax = plt.subplots(figsize=(5.5, 4)) # Slightly adjusted size
        all_scores = []

        # We no longer need 'color_idx'
        for i, (run_set, label) in enumerate(zip(sets, labels)):
            rates, scores = [], []

            for run_id in run_set:
                entry = data.get(run_id, {}).get(ds, {}).get(mdl)
                if not entry:
                    continue
                
                rate = entry.get("orig_bpp") if label.lower() == "original" else entry.get("compressed_bpp")
                
                if not isinstance(category, str):
                    score = sum(entry.get(cat, 0) for cat in category)
                else:
                    score = entry.get(category)

                if rate is not None and score is not None:
                    rates.append(rate)
                    scores.append(score)

            if rates and scores:
                pts = sorted(zip(rates, scores), key=lambda x: x[0])
                r, s = zip(*pts)
                all_scores.extend(s)

                if label.lower() == "original" and len(r) == 1:
                    # This logic for "Original" remains the same
                    ax.hlines(
                        y=s[0],
                        xmin=plot_xlim[0],
                        xmax=plot_xlim[1],
                        linestyles="dotted",
                        label=f"{label}",
                        color="black",
                        linewidth=2,
                    )
                else:
                    # --- 2. MODIFIED: APPLY SEMANTIC STYLES ---
                    current_base = None
                    for base in base_methods:
                        if base in label:
                            current_base = base
                            break
                    
                    # Get styles from maps, with a default
                    current_color = color_map.get(current_base, "black")
                    current_marker = marker_map.get(current_base, "x")
                    
                    # Set linestyle based on your rule
                    current_linestyle = "--" if "Proposed" in label else "-"

                    # --- NEW: Marker fill style and edge properties ---
                    marker_fillstyle = 'none' # Default: filled marker
                    markeredgecolor = current_color # Default: edge same as fill
                    markeredgewidth = 1.8 # Default edge width
                    
                    if "Proposed" in label:
                        marker_fillstyle = 'full' # Make proposed markers hollow
                        markeredgewidth = 1.5 # Slightly thicker edge for hollow markers
                        # markeredgecolor = "black" # Optional: strong black outline
                    # --- END OF NEW SECTION ---
                    
                    ax.plot(
                        r, 
                        s, 
                        marker=current_marker,       # From map
                        linestyle=current_linestyle, # From logic
                        label=label,
                        color=current_color,         # From map
                        linewidth=2,
                        markersize=6,
                        # --- Apply marker properties ---
                        fillstyle=marker_fillstyle,
                        markeredgecolor=markeredgecolor,
                        markeredgewidth=markeredgewidth,
                        # markerfacecolor=current_color if marker_fillstyle == 'full' else 'none' # Redundant with fillstyle
                    )
                    # --- END OF MODIFIED SECTION ---

        # --- Finalize Plot Aesthetics ---
        category_name = category if isinstance(category, str) else "Combined Accuracy"
        
        mdl_name = "InternVL3-9B" if "internvl" in mdl.lower() else "LLaVA-OV-7B" if "llava" in mdl.lower() else mdl
        ds_name = "MMBench" if "mmbench" in ds.lower() else "SEEDBench" if "seedbench" in ds.lower() else ds
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize="large", fontweight='bold')
        ax.set_xlabel("Rate (bpp)", fontsize="medium")
        ax.set_ylabel("Accuracy", fontsize="medium")
        ax.set_xlim(plot_xlim)
        
        if all_scores:
            min_s, max_s = min(all_scores), max(all_scores)
            margin = (max_s - min_s) * 0.05
            # Add a bit of extra top margin for the legend
            ax.set_ylim(min_s - margin, max_s + (margin * 1.5)) 

        # Place legend nicely
        ax.legend(frameon=False, fontsize="small", loc="lower right", handlelength=2.5)
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        sns.despine(ax=ax)
        
        fig.tight_layout()

        save_name = f"rate_accuracy_{ds_name}_{mdl_name}.pdf"
        fig.savefig(
            output_dir / save_name, 
            bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved: {output_dir / save_name}")


def plot_rate_accuracy(
    json_file,
    sets,
    labels,
    pairs,
    category,
    plot_xlim=(0, 0.8),
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

        for run_set, label in zip(sets, labels):
            rates, scores = [], []

            for run_id in run_set:
                entry = data.get(run_id, {}).get(ds, {}).get(mdl)
                if not entry:
                    # skip missing run for this pair
                    continue

                rate = entry.get("orig_bpp") if label == "Original" else entry.get("compressed_bpp")
                if not isinstance(category, str):
                    score = sum(entry.get(cat, 0) for cat in category)
                else:
                    score = entry.get(category)

                if rate is not None and score is not None:
                    rates.append(rate)
                    scores.append(score)

            if rates and scores:
                pts = sorted(zip(rates, scores), key=lambda x: x[0])
                r, s = zip(*pts)

                if label.lower() == "original" and len(r) == 1:
                    # Draw horizontal dashed line from 0 to 1 at score level
                    plt.hlines(
                        y=s[0], xmin=0, xmax=1, colors="C0", linestyles="dashed", label=f"w/o compression", color="gray",
                    )
                else:
                    plt.plot(r, s, marker="o", label=label)

        plt.title(f"{category} — {ds} / {mdl}")
        plt.xlabel("Rate (bpp)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.xlim(plot_xlim)
        plt.savefig(ROOT / f"rate_accuracy_{ds}_{mdl}_{category}_abb_new.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-result-path', type=str, default=str(ROOT / 'results.json'))
    args = parser.parse_args()

    # sets = [
    #     # Original uncompressed
    #     ["prefilter_0-prescale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_0-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M"],

    #     # pure JPEG compression
    #     [
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_5-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_7-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_10-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_15-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",

    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_50-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_60-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_70-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_80-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_JPEG",
    #     ],

    #     # pure HEIF compression
    #     [
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_5-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_7-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_10-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_15-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",

    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_50-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_60-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_70-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_80-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
    #     ],

    #     # pure VVC compression
    #     [
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_50-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_48-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_45-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_42-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_35-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
        
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_25-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_VVC",
    #     ],

    #     # pure Cheng compression
    #     [
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_1-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_2-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_3-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
    #      "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_4-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_5-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
    #     #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-compress_6-cliparch_TinyCLIP_ViT_39M_16_Text_19M-cliptrained_YFCC15M-strictsumm_0-tiles_24-preservesize_1-textsumm_1-compressf_LIC-preprompt_1",
    #     ],

        
    #     # Proposed 22M - JPEG
    #     ["prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_60-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",

    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_70-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-tile_resize-no_aspect",
    #     ],

        
    #     # Proposed 22M - HEVC
    #     [
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",

    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_60-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_70-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
    #     ],

    #     # Proposed 22M - VVC
    #     [
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_48-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_45-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_42-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_35-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_VVC",   
    #     ],

    #     # Proposed 22M - Cheng
    #     [
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_1-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_2-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_3-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
    #      "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_4-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
    #     #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_6-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_LIC-preprompt_1",
    #     ],
    # ]


    # labels = [
    #     "Original",
    #     "JPEG (PIL)",
    #     "HEVC (PIL)",
    #     "VVC (VVenC)",
    #     "Cheng-Anchor",
    #     "Proposed + JPEG",
    #     "Proposed + HEVC",
    #     "Proposed + VVC",
    #     "Proposed + Cheng"
    # ]


    sets = [
        # Original uncompressed
        ["prefilter_0-prescale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_0-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M"],

        # pure HEIF compression
        [
        #  "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_7-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_10-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_15-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
         "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_39M_16_Text_19M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-compressformat_HEIF",
        ],

        # Proposed 57M - HEVC
        [
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
        #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
         "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
        #  "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF",
        ],

        # logitscale=0 - HEVC
        [
        "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        # "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        # "prefilter_1-logitscale_0.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-format_HEIF-preproc_1-bpp",
        ],

        # # TinyCLIP 23M
        # [
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_ViT_8M_16_Text_3M-clippretrained_YFCC15M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_1",
        # ],

        # # TinyCLIP 120M
        # [
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_5-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_7-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_10-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_15-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_20-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_30-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_40-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_50-cliparch_TinyCLIP_auto_ViT_63M_32_Text_31M-cliptrained_LAIONYFCC400M-strictsumm_0-tiles_24-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # ],

        # # Tile_num=4
        # [
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-tiles_4-preservesize_0-textsumm_1-compressf_HEIF-preprompt_1",
        # ],

        # # Proposed 57M - HEVC - original prompt
        # [
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # # "prefilter_1-logitscale_20.0-method_exponential-sigmamin_0.2-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-clippretrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_0-pretextsumm_1-compressformat_HEIF-preprocprompt_0",
        # ]

        # Gaussian 1.0
        [
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_60-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_70-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_1.0-sigmamax_1.0-ksize_11-compress_80-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        ],

        # Gaussian 3.0
        [
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        # "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_60-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_70-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        "prefilter_0-logitscale_0.0-method_exponential-sigmamin_3.0-sigmamax_3.0-ksize_11-compress_80-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_1.0-bpp",
        ],

        # downsample 2
        [
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_60-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_70-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_80-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_2.0-bpp",
        ],
        # downsample 3
        [
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_5-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_7-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_10-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_15-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_20-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_30-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_40-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_50-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_60-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        # "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_70-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        "prefilter_0-logitscale_50.0-method_exponential-sigmamin_0.5-sigmamax_10.0-ksize_11-compress_80-cliparch_TinyCLIP_auto_ViT_22M_32_Text_10M-cliptrained_LAION400M-strictsumm_0-cliptiles_24-preservesize_1-pretextsumm_1-format_HEIF-preproc_1-downsample_3.0-bpp",
        ]
    ]

    labels = [
        "Original",
        "HEVC-Only",
        "Ours-57M",
        "w/o TinyCLIP (Gaussian 0.2)",
        # "Ours-23M",
        # "Ours-120M",
        # "NoOverlap",
        # "w/o PP",
        "Gaussian 1.0",
        "Gaussian 3.0",
        "downsample 2.0",
        "downsample 3.0",
    ]

    pairs = [
    # ("MME", "InternVL3-9B"),
    # ("MMBench_DEV_EN_V11", "InternVL3-9B"),
    # ("SEEDBench_IMG", "InternVL3-9B"),
    # ("MME", "llava-onevision-qwen2-7b-ov-hf"),
    ("MMBench_DEV_EN_V11", "llava-onevision-qwen2-7b-ov-hf"),
    # ("SEEDBench_IMG", "llava-onevision-qwen2-7b-ov-hf"),
    # ("MME", "InternVL2_5-1B"),
    # ("MMBench_DEV_EN_V11", "InternVL2_5-1B"),
    # ("SEEDBench_IMG", "InternVL2_5-1B"),
]

    # category = "Visual Reasoning"
    category = "Overall"
    # category = "perception"
    # category = "reasoning"
    # category = "artwork"
    # category = ["perception", "reasoning"]
    plot_rate_accuracy(args.json_result_path, sets, labels, pairs, category=category, plot_xlim=(0.0, 0.8))
    # plot_rate_accuracy_paper(args.json_result_path, sets, labels, pairs, category=category, plot_xlim=(0, 0.7))
    # plot_rate_accuracy_paper_ablation(args.json_result_path, sets, labels, pairs, category=category, plot_xlim=(0.05, 0.7))


