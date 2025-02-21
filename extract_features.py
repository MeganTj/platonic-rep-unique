import gc
import os
import argparse

from tqdm import trange

import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset
from tasks import get_models
from models import load_llm, load_tokenizer
import json
import utils 
import pdb
from perturbations import text_perturbation_list, image_perturbation_list, save_image

def extract_llm_features(filenames, dataset, args, text_operations=[], text_percentage_perturbation=0.1):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """

    texts = [str(x['text'][args.caption_idx]) for x in dataset]
    if len(text_operations) > 0:
        texts = text_perturbation_list(texts, text_operations, text_percentage_perturbation)

    for llm_model_name in filenames[::-1]:
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, llm_model_name,
            pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx,
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{llm_model_name}")
        print(f'save_path: \t{save_path}')
        
        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue
        
        language_model = load_llm(llm_model_name, qlora=args.qlora, force_download=args.force_download)
        llm_param_count = sum([p.numel() for p in language_model.parameters()])
        tokenizer = load_tokenizer(llm_model_name)
    
        tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
        llm_feats, losses, bpb_losses = [], [], []

        # hack to get around HF mapping data incorrectly when using model-parallel
        device = next(language_model.parameters()).device
        dataset_len = len(dataset)
        for i in trange(0, dataset_len, args.batch_size):
            # get embedding cuda device
            token_inputs = {k: v[i:i+args.batch_size].to(device).long() for (k, v) in tokens.items()}

            with torch.no_grad():
                if "olmo" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                    )

                loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
                losses.extend(avg_loss.cpu())
                
                bpb = utils.cross_entropy_to_bits_per_unit(loss.cpu(), texts[i:i+args.batch_size], unit="byte")
                bpb_losses.extend(bpb)
                
                # make sure to do all the processing in cpu to avoid memory problems
                if args.pool == 'avg':
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif args.pool == 'last':
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2) 
                else:
                    raise NotImplementedError(f"unknown pooling {args.pool}")
                llm_feats.append(feats.cpu())

        print(f"average loss:\t{torch.stack(losses).mean().item()}")
        print(torch.cat(llm_feats).shape)
        save_dict = {
            "feats": torch.cat(llm_feats).cpu(),
            "num_params": llm_param_count,
            "mask": tokens["attention_mask"].cpu(),
            "loss": torch.stack(losses).mean(),
            "bpb": torch.stack(bpb_losses).mean(),
        }

        torch.save(save_dict, save_path)

        del language_model, tokenizer, llm_feats, llm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    return
    
        
def extract_lvm_features(filenames, dataset, args, image_perturbations=None, noise_percentage=0.1, debug_dir=None):
    """
    Extracts features from vision models.
    Args:
        filenames: list of vision model names by timm identifiers
        image_file_paths: list of image file paths
        args: argparse arguments
    """
    assert args.pool == 'cls', "pooling is not supported for lvm features"
    for lvm_model_name in filenames:
        assert 'vit' in lvm_model_name, "only vision transformers are supported"
        
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, lvm_model_name,
            pool=args.pool, prompt=None, caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{lvm_model_name}")
        print(f'save_path: \t{save_path}')

        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue

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
        for i in trange(0, len(dataset), args.batch_size):
            with torch.no_grad():
                max_i = min(len(dataset), i+args.batch_size)
                ims = torch.stack([transform(dataset[j]['image']) for j in range(i, max_i)]).cuda()
                perturbed_ims = ims.clone()  # Make a copy to keep track of perturbed images
                if image_perturbations is not None:
                    perturbed_ims = image_perturbation_list(ims, noise_percentage)

                    # Save original and perturbed images for debugging
                    if debug_dir is not None and i == 0:  # Only save once during the first batch
                        os.makedirs(debug_dir, exist_ok=True)
                        for j in range(min(5, len(perturbed_ims))):  # Save a few images (5 or fewer)
                            # Save the original image
                            save_image(ims[j].cpu(), os.path.join(debug_dir, f"original_{j}_{i}.png"))
                            # Save the perturbed image
                            save_image(perturbed_ims[j].cpu(), os.path.join(debug_dir, f"perturbed_{j}_{i}.png"))

                lvm_output = vision_model(perturbed_ims)

                if args.pool == "cls":
                    feats = [v[:, 0, :] for v in lvm_output.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)
                    
                lvm_feats.append(feats.cpu())
        torch.save({"feats": torch.cat(lvm_feats), "num_params": lvm_param_count}, save_path)

        del vision_model, transform, lvm_feats, lvm_output
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake",   action="store_true")
    parser.add_argument("--num_samples",    type=int, default=1024)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--pool",           type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument("--prompt",         action="store_true")
    parser.add_argument("--dataset",        type=str, default="minhuh/prh")
    parser.add_argument("--subset",         type=str, default="wit_1024")
    parser.add_argument("--caption_idx",    type=int, default=0)
    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--modality",       type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir",     type=str, default="./results/features")
    parser.add_argument("--debug_dir",     type=str, default=None)
    parser.add_argument("--qlora",          action="store_true")
    # Noise Arguments
    parser.add_argument(
        "--text_operations",
        nargs="+",  # This allows multiple values
        type=str,   # Ensure values are treated as strings
        help="List of strings to process",
        default=[]
    )
    parser.add_argument("--text_percentage_perturbation", type=float, default=0.1)
    parser.add_argument(
        "--image_perturbations",
        default=None)
    parser.add_argument("--noise_percentage", type=float, default=0.1)

    args = parser.parse_args()

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    llm_models, lvm_models = get_models(args.modelset, modality=args.modality)
    
    # load dataset once outside    
    dataset = load_dataset(args.dataset, revision=args.subset, split='train')
    if args.modality in ["all", "language"]:
        # extract all language model features
        extract_llm_features(llm_models, dataset, args, text_operations=args.text_operations, text_percentage_perturbation=args.text_percentage_perturbation)
    
    if args.modality in ["all", "vision"]:
        # extract all vision model features
        extract_lvm_features(lvm_models, dataset, args, image_perturbations=args.image_perturbations, noise_percentage=args.noise_percentage, debug_dir=args.debug_dir)
