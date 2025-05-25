import matplotlib.pyplot as plt
import argparse
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import metrics
import utils
import numpy as np
import torch
from tasks import get_models
from scipy import stats
from sklearn.linear_model import LinearRegression
from dataset import genres_
from collections import defaultdict
# from mmimdb import NUM_GENRES

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

def plot_helper(perf, size_scores, ax, colors, title, ylim=None, size_idx=None):
    all_r = []
    all_slope = []
    if size_idx is not None:
        size_scores = dict([list(size_scores.items())[-1]])
    for size_idx, (size, scores) in enumerate(size_scores.items()):
        color = colors[size_idx % len(colors)]
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.scatter(
            perf, 
            scores, 
            label=size, 
            color=color, 
            edgecolor='k',  # Add edge color to markers
            s=80,  # Increase marker size
            alpha=0.8  # Slightly transparent markers
        )
        rho, slope, intercept = compute_spearman_and_regression(perf, scores)
        ax.plot(perf, intercept + slope * np.array(perf), 
                color=color)
        all_slope.append(slope)
        all_r.append(rho)
        
    mean_r = np.array(all_r).mean()
    mean_slope = np.array(all_slope).mean()
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set titles and labels
    ax.set_title(f"{title}, slope={mean_slope:.3f}, $\\rho$={mean_r:.3f}", fontsize=SMALL_TITLE_FONTSIZE)
    ax.set_xlabel("F1-Score", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f"Alignment to {VISION_KEYWORD_TO_PLOT[keyword]}", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return mean_r, mean_slope


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
    parser.add_argument("--perf_dir",      type=str, default="/scratch/platonic/mmimdb/mmimdb_performance_upd/")
    parser.add_argument("--perf_filename", type=str, default="test_perf.npy")
    parser.add_argument("--align_dir",     type=str, default="/scratch/platonic/mmimdb/mmimdb_align")
    parser.add_argument("--same_scale",     action="store_true")
    parser.add_argument("--plot_dir",     type=str, default="./mmimdb/plots")
    args = parser.parse_args()

    llm_models, lvm_models = get_models(args.modelset, modality='all')
    llm_plot_names = [convert_model_name_to_plot(model_name) for model_name in llm_models]
    os.makedirs(args.plot_dir, exist_ok=True)
    # keyword -> size -> list over uniqueness 
    keywords = ["dino", "clip", "mae", "augreg", "ft_in12k"]
    all_scores = {keyword: {} for keyword in keywords}
    for keyword in keywords:
        sizes_idx = parse_size(lvm_models, keyword)
        for size, _ in sizes_idx:
            # Initialize list of alignment scores to different language models
            all_scores[keyword][size] = []

    align_path = utils.to_alignment_filename(
                args.align_dir, args.modelset,
                args.metric, args.topk
    )
    align_scores = np.load(align_path, allow_pickle=True).item()['scores']
    
    llm_perf = []
    # Iterate over all model subdirectories in perf_dir
    for i, llm_model in enumerate(llm_models):
        test_perf_dir = utils.to_model_savedir(args.perf_dir, llm_model)
        test_perf_file = os.path.join(test_perf_dir, args.perf_filename)
        test_metrics = np.load(test_perf_file, allow_pickle=True).item()
        llm_perf.append(test_metrics)

        for keyword in keywords:
            sizes_idx = parse_size(lvm_models, keyword)
            for size, j in sizes_idx:
                align_score = align_scores[i][j]
                all_scores[keyword][size].append(align_score) 
    #  Use a professional color palette (e.g., tab10)
    colors = plt.cm.tab10.colors
    # Plot alignment w.r.t. different downstream tasks 
    overall_perf = [metrics["f1_mean"] for metrics in llm_perf]
    all_class_perf = [metrics["f1_per_class"] for metrics in llm_perf]
    plot_genres = None
    all_keyword_r = []
    all_keyword_slope = []
    for keyword, size_scores in all_scores.items():
        fig, ax = plt.subplots()
        plot_helper(overall_perf, size_scores, ax, colors, "Overall Performance")
        save_path = os.path.join(args.plot_dir, f"{keyword}_align_perf_overall")
        plt.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
        plt.savefig(save_path + ".pdf", bbox_inches='tight')
        plt.close()
        all_r = []
        all_slope = []
        for class_idx, genre in enumerate(genres_):
            class_plot_dir = os.path.join(args.plot_dir, keyword)
            os.makedirs(class_plot_dir, exist_ok=True)
            fig, ax = plt.subplots()
            class_perf = [perf[class_idx] for perf in all_class_perf]
            class_r, class_slope = plot_helper(class_perf, size_scores, ax, colors, f"Performance on {genre}")
            all_r.append(np.array(class_r))
            all_slope.append(np.array(class_slope))
            save_path = os.path.join(class_plot_dir, f"{keyword}_align_perf_{genre}")
            plt.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
            plt.savefig(save_path + ".pdf", bbox_inches='tight')
            plt.close()
        all_keyword_r.append(np.array(all_r))
        all_keyword_slope.append(np.array(all_slope))

    save_dct = {"corr": np.array(all_keyword_r).mean(0), "slope": np.array(all_keyword_slope).mean(0)}

    size_mult = 5
    max_r = -np.inf
    max_slope = -np.inf
    for keyword_r, keyword_slope in zip(all_keyword_r, all_keyword_slope):
        max_r = max(max_r, np.histogram(keyword_r, bins=10)[0].max())  # Find the max frequency across all histograms
        slope_hist = np.histogram(keyword_slope, bins=10)[0]
        max_slope = max(max_slope, slope_hist.max())
        
    fig, axs = plt.subplots(1, 5, figsize=(5 * size_mult, 1 * size_mult))
    # Plot the distribution of r and slope
    for axs_idx, keyword_r in enumerate(all_keyword_r):
        # Customizing the plot
        axs[axs_idx].set_facecolor(BACKGROUND_COLOR)
        axs[axs_idx].hist(keyword_r, bins=10, edgecolor='black', alpha=0.7)
        # Customizing the plot
        axs[axs_idx].set_title(f'{VISION_KEYWORD_TO_PLOT[keywords[axs_idx]]}', fontsize=SMALL_TITLE_FONTSIZE)
        axs[axs_idx].set_xlabel('Spearman $\\rho$', fontsize=SMALL_AXIS_LABEL_FONTSIZE)
        axs[axs_idx].set_ylabel('Frequency', fontsize=SMALL_AXIS_LABEL_FONTSIZE)
        axs[axs_idx].grid(True, linestyle='--', alpha=0.6)
        axs[axs_idx].tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
        # Set the same y-axis limit across all subplots
        axs[axs_idx].set_ylim(0, max_r + 0.5)
    save_path = os.path.join(args.plot_dir, f"align_perf_corr")
    plt.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
    plt.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(1, 5, figsize=(5 * size_mult, 1 * size_mult))
    # Plot the distribution of r and slope
    for axs_idx, keyword_slope in enumerate(all_keyword_slope):
        # Customizing the plot
        axs[axs_idx].set_facecolor(BACKGROUND_COLOR)
        axs[axs_idx].hist(keyword_slope, bins=10, edgecolor='black', alpha=0.7)
        axs[axs_idx].set_title(f'{VISION_KEYWORD_TO_PLOT[keywords[axs_idx]]}', fontsize=SMALL_TITLE_FONTSIZE)
        axs[axs_idx].set_xlabel('Fitted Slopes', fontsize=SMALL_AXIS_LABEL_FONTSIZE)
        axs[axs_idx].set_ylabel('Frequency', fontsize=SMALL_AXIS_LABEL_FONTSIZE)
        axs[axs_idx].grid(True, linestyle='--', alpha=0.6)
        axs[axs_idx].tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
        # Set the same y-axis limit across all subplots
        axs[axs_idx].set_ylim(0, max_slope + 0.5)
    save_path = os.path.join(args.plot_dir, f"align_perf_slope")
    plt.savefig(save_path + ".png", bbox_inches='tight', dpi=300)
    plt.savefig(save_path + ".pdf", bbox_inches='tight')
    plt.close()

    # Get the slopes for 10 classes
    curr_keyword_slope = all_keyword_slope[0]
    final_class_idx = [13, 19, 10, 6, 2, 15, 1, 16, 0, 17]
    class_corr = {class_idx: curr_keyword_slope[class_idx] for class_idx in final_class_idx}
    np.save(os.path.join(args.plot_dir, "class_corr.npy"), class_corr)
        
