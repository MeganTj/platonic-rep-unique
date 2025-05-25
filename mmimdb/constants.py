import os

ROOT = "/scratch/megantj/datasets/mmimdb"
SAMPLE_FORMAT = os.path.join(ROOT, "{split}/sampled_ids.npy")
DATA_SPLITS = ["train", "dev", "test"]
def get_sampled_save_path(split):
    return os.path.join(ROOT, split, "sampled_ids.npy")