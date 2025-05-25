import os
import argparse 

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
import utils
from pprint import pprint
from extract_features import to_feature_filename
from measure_alignment import compute_alignment

if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",           type=int, default=10)

    parser.add_argument("--feat_save_dir",      type=str, default="/scratch/platonic/mmimdb/mmimdb_feat")
    parser.add_argument("--output_dir",     type=str, default="/scratch/platonic/mmimdb/mmimdb_align")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    args = parser.parse_args()
    
    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    save_path = utils.to_alignment_filename(
            args.output_dir, args.modelset,
            args.metric, args.topk
    )
    
    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()
    
    llm_models, lvm_models = get_models(args.modelset, modality='all')
    
    models_x_paths = [to_feature_filename(args.feat_save_dir, model_name, "test") for model_name in llm_models]
    models_y_paths = [to_feature_filename(args.feat_save_dir, model_name, "test") for model_name in lvm_models]
    
    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn
    
    print(f"metric: \t{args.metric}")
    if 'knn' in args.metric:
        print(f"topk:\t{args.topk}")
    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')
    alignment_scores, alignment_indices = compute_alignment(args, models_x_paths, models_y_paths, args.metric, args.topk, args.precise)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"scores": alignment_scores, "indices": alignment_indices})
    print(f"saved to {save_path}")
    