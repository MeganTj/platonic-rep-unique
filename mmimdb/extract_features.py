import argparse
import torch
from tqdm import trange
import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
from constants import ROOT, SAMPLE_FORMAT
from models import load_llm, load_tokenizer
from tasks import get_models

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image

def get_llm_embeddings(tokens, language_model):
    with torch.no_grad():
        outputs = language_model(**tokens)
    # You can use the last hidden state, or pool the embeddings (mean/CLS/etc.)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling to get a single vector per sentence
    return embeddings

def to_feature_filename(output_dir, model_name, split, pool=None, prompt=None, caption_idx=None):
    save_name = f"{model_name.replace('/', '_')}_{split}"

    if pool:
        save_name += f"_pool-{pool}"
    if prompt:
        save_name += f"_prompt-{prompt}"
    if caption_idx:
        save_name += f"_cid-{caption_idx}"
    
    save_path = os.path.join(
        output_dir, 
        f"{save_name}.pt"
    )
    return save_path

def extract_llm_feats(args, llm_model_name, data_splits, captions, device):
    print(f"processing:\t{llm_model_name}")
    base_model = None
    tokenizer = None
    for data_split, captions in zip(data_splits, captions):
        save_path = to_feature_filename(args.feat_save_dir, model_name, data_split)
        print(f'save_path: \t{save_path}')
        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue
        if base_model is None:
            base_model = load_llm(model_name, qlora=False)
            tokenizer = load_tokenizer(model_name)
        tokens = tokenizer(captions, padding="longest", return_tensors="pt") 
        llm_param_count = sum([p.numel() for p in base_model.parameters()])
        llm_feats = []
        for i in trange(0, len(captions), args.batch_size): 
            token_inputs = {k: v[i:i+args.batch_size].to(device).long() for (k, v) in tokens.items()}  
            for i, input_ids in enumerate(token_inputs["input_ids"]):
                if (input_ids == tokenizer.unk_token_id).any():
                    print("There are out-of-vocabulary tokens.")
            with torch.no_grad():
                if "olmo" in llm_model_name.lower():
                    llm_output = base_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    llm_output = base_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                    )
            
                if torch.stack(llm_output["hidden_states"]).isnan().any():
                    import pdb
                    pdb.set_trace()
                #  make sure to do all the processing in cpu to avoid memory problems
                # Average over tokens
                feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                feats_avg = (feats * mask).sum(2) / mask.sum(2)
                llm_feats.append(feats_avg.cpu())
        save_dict = {
            "feats": torch.cat(llm_feats).cpu(),
            "num_params": llm_param_count,
        }
        torch.save(save_dict, save_path)
        del llm_feats, llm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    del base_model, tokenizer

def extract_lvm_features(args, lvm_model_name, save_path, imgs, device):
    """
    Extracts features from vision models.
    Args:
        filenames: list of vision model names by timm identifiers
        image_file_paths: list of image file paths
        args: argparse arguments
    """
    assert 'vit' in lvm_model_name, "only vision transformers are supported"
    
    print(f"processing:\t{lvm_model_name}")
    print(f'save_path: \t{save_path}')

    if os.path.exists(save_path) and not args.force_remake:
        print("file exists. skipping")
        return

    vision_model = timm.create_model(lvm_model_name, pretrained=True).cuda().eval()
    lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

    transform = create_transform(
        **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
    )

    if "vit" in lvm_model_name:
        return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
    else:
        raise NotImplementedError(f"unknown model {lvm_model_name}")

    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
    lvm_feats = []
    for i in trange(0, len(imgs), args.batch_size):
        with torch.no_grad():
            max_i = min(len(imgs), i+args.batch_size)
            batch_imgs = torch.stack([transform(imgs[j]) for j in range(i, max_i)]).cuda()

            lvm_output = vision_model(batch_imgs)

            # Pool over class tokens
            feats = [v[:, 0, :] for v in lvm_output.values()]
            feats = torch.stack(feats).permute(1, 0, 2)
                
            lvm_feats.append(feats.cpu())
    torch.save({"feats": torch.cat(lvm_feats), "num_params": lvm_param_count}, save_path)

    del vision_model, transform, lvm_feats, lvm_output
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_img_captions(root, split, sampled_ids=None, return_captions_only=False):
    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    metadata = os.path.join(root, "split.json")
    if sampled_ids is not None:
        ids = np.load(sampled_ids)
    else:
        with open(metadata) as f:
            ids = json.load(f)[split]
    imgs = []
    captions = []
    for img_id in ids:
        sample_path = os.path.join(root, 'dataset', f'{img_id}.json')
        with open(sample_path) as f:
            meta = json.load(f)
            path = os.path.join(root, 'dataset', '{}.jpeg'.format(img_id))
            if not return_captions_only:
                img = pil_loader(path)
                imgs.append(img)
            for caption in meta['plot']:
                if len(caption.split(' ')) > 1:
                    break
            assert len(caption.split(' ')) > 1
            captions.append(caption)
    return imgs, captions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--force_remake",   action="store_true")
    parser.add_argument("--feat_save_dir", type=str, default="/scratch/platonic/mmimdb/mmimdb_feat")
    args = parser.parse_args()
    device = "cuda:0"
    os.makedirs(args.feat_save_dir, exist_ok=True)
    # Create DataLoader for validation and test sets
    data_splits = ["train", "dev", "test"]
    _, train_captions = load_img_captions(ROOT, "train", sampled_ids=SAMPLE_FORMAT.format(split="train"), return_captions_only=True)
    _, val_captions = load_img_captions(ROOT, "dev", sampled_ids=SAMPLE_FORMAT.format(split="dev"), return_captions_only=True)
    test_imgs, test_captions = load_img_captions(ROOT, "test", sampled_ids=SAMPLE_FORMAT.format(split="test"))

    llm_models, lvm_models = get_models(args.modelset, modality="all")
    # Extract features
    for model_name in llm_models[::-1]:
        extract_llm_feats(args, model_name, data_splits, [train_captions, val_captions, test_captions], device)


    for model_name in lvm_models:
        feat_save_path = to_feature_filename(args.feat_save_dir, model_name, "test")
        extract_lvm_features(args, model_name, feat_save_path, test_imgs, device)