import os

def to_model_savedir(output_dir, model_name):
    save_name = f"{model_name.replace('/', '_')}"
    
    save_dir = os.path.join(
        output_dir, 
        save_name
    )
    return save_dir

def to_alignment_filename(output_dir, modelset, 
                          metric, topk):
    save_path = os.path.join(
        output_dir,
        modelset,
        f"{metric}_k{topk}.npy" if 'knn' in metric else f"{metric}.npy"
    )
    return save_path
    