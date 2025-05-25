import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from constants import SAMPLE_FORMAT
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def compute_pos_weights_binary(downstream_data_module):
    # Get train labels from the DataLoader
    train_dataloader = downstream_data_module.train_dataloader()  # This should return your DataLoader
    
    # Initialize an empty list to store all the labels
    train_labels = []

    # Iterate over the DataLoader to collect labels
    for batch in train_dataloader:
        # Assuming batch is a tuple (data, labels) and labels are the second element
        labels = batch[1]  # Adjust based on your DataLoader format
        train_labels.append(labels)
    
    # Convert train_labels to a tensor for easier manipulation
    train_labels = torch.cat(train_labels)
    assert torch.all(train_labels >= 0)
    # Count the number of positive labels for each class (column-wise sum)
    label_sums = train_labels.sum(axis=0)  # Sum over samples for each label
    
    # Compute the inverse of class frequency as the weight for the positive class
    total_samples = train_labels.shape[0]
    pos_weight = (total_samples - label_sums) / label_sums
    pos_weight = torch.tensor(pos_weight, dtype=torch.float)
    
    return pos_weight



@hydra.main(version_base=None, config_name="train_mmimdb", config_path="../configs")
def main(cfg: DictConfig):
    """
    Training/test of Multi-Modal models on MM-IMDB dataset.
    Models currently implemented are:
        - CLIP

    """
    # fix the seed for repro
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Data loading code
    use_sup = False
    if cfg.model.name == "CLIPSup":
        use_sup = cfg.model.model.clip_weight == 0
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name,
                            sample_format=SAMPLE_FORMAT, class_idx=cfg.class_idx,
                            use_sup=use_sup)
    # create model + save hyper-parameters
    dataset = "mmimdb"
    model_kwargs = dict()

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **model_kwargs)

    model.save_hyperparameters(cfg)


    if cfg.class_idx is not None:
        cfg.eval_class_idx = [cfg.class_idx]
    # Train a single model, potentially have multiple downstream data modules
    downstream_data_modules = []
    names = []
    # Create a list to store model checkpoint callbacks for each class
    checkpoint_callbacks = []
    for class_idx in cfg.eval_class_idx:
        downstream_data_modules.append(instantiate(cfg.data.data_module, model="Sup", 
                                                   sample_format=SAMPLE_FORMAT, class_idx=class_idx))
        dataset_name = f"{dataset}.{class_idx}"
        names.append(f"{dataset}.{class_idx}")
        checkpoint_callbacks.append(
            ModelCheckpoint(monitor=f'test_f1_mean_{dataset_name}', mode='max', save_top_k=1)
        )

    logger = TensorBoardLogger(build_root_dir(cfg), name="logs")
    callbacks = [instantiate(cfg.linear_probing, downstream_data_modules=downstream_data_modules, 
                            names=names, multilabel=False, include_dataset=True), 
                *checkpoint_callbacks,]
    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir = build_root_dir(cfg),
        logger=logger,
        callbacks=callbacks)
    test_results = trainer.test(model, downstream_data_modules[0])
    print(test_results)
    trainer = instantiate(
        cfg.trainer,
        default_root_dir = build_root_dir(cfg),
        logger=logger,
        callbacks=callbacks)
    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        raise NotImplementedError


def build_root_dir(cfg: DictConfig):
    # set directory for logs and checkpoints
    if "Sup" in cfg.model.name:
        hyperparam_str = f"clip-weight_{cfg.model.model.clip_weight}/class_{cfg.class_idx}"
        root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, hyperparam_str, "mmimdb")
    else:
        root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, "mmimdb")
    print(root_dir)
    # modify `root_dir` if in test mode to match pre-trained model's path
    if cfg.mode == "test":
        if getattr(cfg, "ckpt_path", None) is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(cfg.ckpt_path), "test")

    if getattr(cfg, "exp_name", None) is not None:
        root_dir = os.path.join(root_dir, cfg.exp_name)

    return root_dir


if __name__ == '__main__':
     main()