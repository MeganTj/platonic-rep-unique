import matplotlib.pyplot as plt
import argparse
import os
import pdb
import metrics
import utils
import numpy as np
import torch
from tasks import get_models
from scipy import stats
from sklearn.linear_model import LinearRegression
from collections import defaultdict

AXIS_LABEL_FONTSIZE = 18
TICK_SIZE=16
TITLE_FONTSIZE=20
SMALL_AXIS_LABEL_FONTSIZE = 14
SMALL_TICK_SIZE=12
SMALL_TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 10
SUPTITLE_FONTSIZE = 22
# Define the background color (e.g., light gray)
BACKGROUND_COLOR = '#f7f7f7'  # Light gray hex color

LLM_RAW_NAME_TO_PLOT = {
    "bloomz-": "bloom",
    "open_llama_": "openllama",
    "llama-": "llama",
    "560m": "0.56b",
    "1b1": "1.1b",
    "1b7": "1.7b",
    "3b": "3b",
    "7b1": "7b",
}

VISION_KEYWORD_TO_PLOT = {
    "dino": "DINOv2",
    "augreg": "ImageNet21k",
    "clip": "CLIP",
    "ft_in12k": "CLIP (I12K ft)",
    "mae": "MAE",
}

def parse_size(model_paths, filter_keyword):
    sizes = []
    for idx, path in enumerate(model_paths):
        if filter_keyword in path:
            if filter_keyword != "clip" or "ft_in12k" not in path:
                start_idx = path.index("vit_")
                end_idx = path.index("_", start_idx + 4)
                sizes.append((path[start_idx+4:end_idx], idx))
    return sizes

def convert_model_name_to_plot(model_name):
    base_name = os.path.basename(model_name)
    for raw_str, plot_str in LLM_RAW_NAME_TO_PLOT.items():
        base_name = base_name.replace(raw_str, plot_str)
    return base_name

# Function to compute Spearman's rho, slope, and intercept of the regression line
def compute_spearman_and_regression(x, y):
    # Compute Spearman's rho (monotonicity strength) between ranks of x and y
    rho, _ = stats.spearmanr(x, y)
    
    # Reshape x for linear regression (x as independent variable)
    X = np.array(x).reshape(-1, 1)  # x-values (independent)
    Y = np.array(y)  # y-values (dependent)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, Y)
    
    # Get the slope and intercept of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return rho, slope, intercept


if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="prh/minhuh")
    parser.add_argument("--subset",         type=str, default="wit_1024")

    parser.add_argument("--modality_x",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_x",       action="store_true")
    parser.add_argument("--pool_x",         type=str, default=None, choices=['avg', 'cls'])
    
    parser.add_argument("--modality_y",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_y",       action="store_true")
    parser.add_argument("--pool_y",         type=str, default=None, choices=['avg', 'cls'])

    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",           type=int, default=10)
    parser.add_argument("--same_scale",     action="store_true")
    parser.add_argument("--features_dir",      type=str, default="/scratch/platonic/results/u={}_features")
    parser.add_argument("--align_dir",     type=str, default="/scratch/platonic/results/u={}_alignment")
    parser.add_argument("--plot_dir",     type=str, default="./plots")
    args = parser.parse_args()

    min_perturbation = 10
    max_perturbation = 100
    step = 10
    all_perturbations = np.arange(min_perturbation, max_perturbation + 1, step)
    llm_models, lvm_models = get_models(args.modelset, modality='all')
    models_x = llm_models if args.modality_x == "language" else lvm_models
    models_y = llm_models if args.modality_y == "language" else lvm_models
    llm_plot_names = [convert_model_name_to_plot(model_name) for model_name in llm_models]
    os.makedirs(args.plot_dir, exist_ok=True)
    # keyword -> size -> list over uniqueness 
    keywords = ["dino", "clip", "mae", "augreg", "ft_in12k"]
    all_scores = {keyword: {} for keyword in keywords}
    min_align = {keyword: np.inf for keyword in keywords}
    max_align = {keyword: -np.inf for keyword in keywords}
    for keyword in keywords:
            sizes_idx = parse_size(lvm_models, keyword)
            for size, _ in sizes_idx:
                # Initialize list over uniqueness 
                all_scores[keyword][size] = [[] for _ in all_perturbations]
    all_perf = [[] for _ in all_perturbations]
    for unique_idx, perturbation_val in enumerate(all_perturbations):
        align_path = utils.to_alignment_filename(
                args.align_dir, args.dataset, args.modelset,
                args.modality_x, args.pool_x, args.prompt_x,
                args.modality_y, args.pool_y, args.prompt_y,
                args.metric, args.topk
        ).format(perturbation_val)
        align_scores = np.load(align_path, allow_pickle=True).item()['scores']
        features_dir = args.features_dir.format(perturbation_val)
        models_x_paths = [utils.to_feature_filename(features_dir, args.dataset, args.subset, m, args.pool_x, args.prompt_x) for m in models_x]
        models_y_paths = [utils.to_feature_filename(features_dir, args.dataset, args.subset, m, args.pool_y, args.prompt_y) for m in models_y]
        if args.modality_x == "language":
            language_paths = models_x_paths
            vision_paths = models_y_paths
        else:
            vision_paths = models_x_paths
            language_paths = models_y_paths 
        for i, x_fp in enumerate(language_paths):
            feats_dct = torch.load(x_fp, map_location="cuda:0")
            all_perf[unique_idx].append((-feats_dct["bpb"]).cpu().numpy())
            for keyword in keywords:
                sizes_idx = parse_size(vision_paths, keyword)
                for size, j in sizes_idx:
                    if args.modality_x == "language":
                        align_score = align_scores[i][j]
                        all_scores[keyword][size][unique_idx].append(align_score) 
                    else:
                        align_score = align_scores[j][i]
                        all_scores[keyword][size][unique_idx].append(align_score) 
                    if align_score < min_align[keyword]:
                        min_align[keyword] = align_score
                    if align_score > max_align[keyword]:
                        max_align[keyword] = align_score
                    

    for keyword, size_scores in all_scores.items():
        size_mult = 4
        fig, axs = plt.subplots(1, 5, figsize=(5 * size_mult, 1 * size_mult))
        axs = axs.flatten()
        # Use a professional color palette (e.g., tab10)
        colors = plt.cm.tab10.colors
        for idx in range(0, len(all_perturbations), 2):
            perturbation_val = all_perturbations[idx]
            all_r = []
            all_slope = []
            plot_idx = idx // 2
            axs[plot_idx].set_facecolor(BACKGROUND_COLOR)
            for size_idx, (size, unique_scores) in enumerate(size_scores.items()):
                # Scatter plot
                color = colors[size_idx % len(colors)]
                axs[plot_idx].scatter(
                    all_perf[idx], 
                    unique_scores[idx], 
                    label=size, 
                    color=color, 
                    edgecolor='k',  # Add edge color to markers
                    s=80,  # Increase marker size
                    alpha=0.8  # Slightly transparent markers
                )
                rho, slope, intercept = compute_spearman_and_regression(all_perf[idx], unique_scores[idx])
                axs[plot_idx].plot(all_perf[idx], intercept + slope * np.array(all_perf[idx]), 
                        color=color)
                all_slope.append(slope)
                all_r.append(rho)
            mean_slope = np.array(all_slope).mean()
            mean_r = np.array(all_r).mean()
        
            # Add grid lines
            axs[plot_idx].grid(True, linestyle='--', alpha=0.6)
            
            # Set titles and labels
            axs[plot_idx].set_title(f"$U$={perturbation_val}, slope={mean_slope:.3f}", fontsize=SMALL_TITLE_FONTSIZE)
            axs[plot_idx].set_xlabel("Performance", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
            axs[plot_idx].set_ylabel(f"Alignment to {VISION_KEYWORD_TO_PLOT[keyword]}", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
            axs[plot_idx].tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
            
            if args.same_scale:
                axs[plot_idx].set_ylim(min_align[keyword], max_align[keyword])
            # Add legend
            axs[plot_idx].legend(fontsize=LEGEND_FONTSIZE, loc='upper left', framealpha=0.5)

        # Adjust layout
        fig.tight_layout()

        # Save the plot
        save_path = os.path.join(args.plot_dir, f"{keyword}_align_perf_unique")
        plt.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
        plt.savefig(save_path + ".pdf", bbox_inches='tight')
        plt.close()
        plt.figure()

        # Plot alignment vs uniqueness for each size
        plt.figure(figsize=(6, 5))

        # Set constant background shading
        plt.gca().set_facecolor(BACKGROUND_COLOR)

        # Use a professional color palette (e.g., tab10)
        colors = plt.cm.tab10.colors

        # Plot the data
        all_res = []
        for size_idx, (size, unique_scores) in enumerate(size_scores.items()):
            all_u = []
            all_align = []
            for idx, perturbation_val in enumerate(all_perturbations):
                all_u.extend([perturbation_val for _ in range(len(unique_scores[idx]))])
                all_align.extend(unique_scores[idx])
            # res = stats.linregress(all_u, all_align)
            rho, slope, intercept = compute_spearman_and_regression(all_u, all_align)
            all_res.append(rho)
            # Scatter plot
            plt.scatter(
                all_u, 
                all_align, 
                label=size, 
                color=colors[size_idx % len(colors)], 
                edgecolor='k',  # Add edge color to markers
                s=80,  # Increase marker size
                alpha=0.8  # Slightly transparent markers
            )
        mean_r = np.array(all_res).mean()
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.title(f"{VISION_KEYWORD_TO_PLOT[keyword]}, $\\rho$={mean_r:.3f}", fontsize=TITLE_FONTSIZE)
        # Set titles and labels
        plt.xlabel("Unique", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("Alignment", fontsize=AXIS_LABEL_FONTSIZE)
        plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Add legend
        plt.legend(fontsize=LEGEND_FONTSIZE, loc='upper right', framealpha=1.0)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(args.plot_dir, f"{keyword}_align_unique")
        plt.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
        plt.savefig(save_path + ".pdf", bbox_inches='tight')
        plt.close()
        
