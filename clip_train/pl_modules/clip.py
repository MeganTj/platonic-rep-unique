# Modified from github.com/facebookresearch/SLIP
import numpy as np
import torch
import math
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from torch import nn
from typing import Dict
from pl_modules.base import BaseModel
from losses.clip_loss import CLIPLoss, BCELossWithMasking


class CLIP(BaseModel):
    """CLIP model for Vision-Language representation learning. """
    def __init__(self,
                 visual: nn.Module,
                 language: nn.Module,
                 optim_kwargs: Dict,
                 image_projection: nn.Module = None,
                 text_projection: nn.Module = None):
        """
        Args:
            visual: Vision encoder (e.g. ViT or ResNet50)
            language: Text encoder (e.g. Transformer)
            optim_kwargs: Optimization hyper-parameters to train CLIP
            image_projection: linear projector to the embedding space (optional)
            text_projection: linear projection to the embedding space (optional)
        """
        super().__init__(optim_kwargs)

        self.visual = visual
        self.language = language

        self.image_projection = image_projection or nn.Identity()
        self.text_projection = text_projection or nn.Identity()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss = CLIPLoss()
        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.image_projection, nn.Linear):
            vision_width = self.image_projection.weight.shape[1]
            nn.init.normal_(self.image_projection.weight, std=vision_width ** -0.5)
        if isinstance(self.text_projection, nn.Linear):
            transformer_width = self.text_projection.weight.shape[1]
            nn.init.normal_(self.text_projection.weight, std=transformer_width ** -0.5)

    def encode_image(self, image):
        x = self.visual(image)
        x = self.image_projection(x)
        return x

    def encode_text(self, text):
        x = self.language(text)
        x = self.text_projection(x)
        return x

    def training_step(self, batch, batch_idx):
        # Labels are provided
        inputs = batch
        if len(batch) == 2:
            inputs = batch[0]
        outputs = self.forward(*inputs)
        out_dict = self.loss(outputs)
        loss = out_dict['loss']
        self.logit_scale.data.clamp_(0, 4.6052)
        if not math.isfinite(loss.item()):
            print("Loss is {}, logit scale is {}, stopping training".format(loss.item(), self.logit_scale))
            sys.exit(1)
        self.log_dict(out_dict, on_epoch=True, sync_dist=True)
        return loss

    def extract_features(self, loader: torch.utils.data.DataLoader,
                         encode_text: bool = True,
                         encode_image: bool = True,
                         separate_modalities=False):
        """
           Extract CLIP features (from vision, language or both)
           Args:
               loader: Dataset loader to serve ``(X, y)`` tuples.
               encode_text: If true, encodes text modality
               encode_image: If true, encodes image modality
                    If both `encode_image` and `encode_text` are true,
                    returns a concatenation of image+text features (in that order)
            Returns: Pair (X,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        if separate_modalities:
            X= [[], []]
        for X_, y_ in loader:
            images, text = None, None
            if isinstance(X_, list): # first modality == image, second modality == text (convention)
                if encode_image:
                    images = X_[0].to(self.device)
                if encode_text:
                    text = X_[1]
                    if isinstance(text, torch.Tensor):
                        text = text.to(self.device)
            else:
                assert encode_image ^ encode_text, \
                    "Unknown input modality: `encode_text` or `encode_image` must be specified"
                if encode_image:
                    images = X_.to(self.device)
                if encode_text:
                    text = X_.to(self.device)
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                output = []
                if images is not None:
                    output.append(self.encode_image(images).view(len(images), -1))
                if text is not None:
                    output.append(self.encode_text(text).view(len(text), -1))
                if separate_modalities:
                    X[0].extend(output[0].detach().cpu())
                    X[1].extend(output[1].detach().cpu())
                else:
                    output = torch.cat(output, dim=-1)
                    X.extend(output.detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        if separate_modalities:
            return (torch.stack(X[0], dim=0).to(self.device), torch.stack(X[1], dim=0).to(self.device)), torch.stack(y, dim=0).to(self.device)
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}

class CLIPSup(CLIP):
     def __init__(self,
                 visual: nn.Module,
                 language: nn.Module,
                 optim_kwargs: Dict,
                 image_projection: nn.Module = None,
                 text_projection: nn.Module = None,
                 text_input_dim=None,
                 pos_weight=None,
                 clip_weight=1.0,
                 sup_only=False):
        """
        Args:
            visual: Vision encoder (e.g. ViT or ResNet50)
            language: Text encoder (e.g. Transformer)
            optim_kwargs: Optimization hyper-parameters to train CLIP
            image_projection: linear projector to the embedding space (optional)
            text_projection: linear projection to the embedding space (optional)
        """
        super().__init__(visual, language, optim_kwargs, image_projection, text_projection)
        self.supervised_loss = BCELossWithMasking(pos_weight=pos_weight)
        self.text_prediction_head = nn.Linear(text_input_dim, 1)
        self.clip_weight = clip_weight
        # self.sup_only = sup_only

     def compute_loss(self, outputs, batch):
        """
        Compute the loss for the given outputs.
        Args:
            outputs: The outputs from the forward pass of the model.
        """
        text_predictions = self.text_prediction_head(outputs["text_embed"])
        sup_loss = self.supervised_loss(text_predictions, batch[1])
        if self.clip_weight > 0:
            out_dict = self.loss(outputs)

            clip_loss = out_dict['loss']
            self.logit_scale.data.clamp_(0, 4.6052)
            if not math.isfinite(clip_loss.item()):
                print("Loss is {}, logit scale is {}, stopping training".format(clip_loss.item(), self.logit_scale))
                sys.exit(1)
            out_dict["sup_loss"] = sup_loss
            out_dict["clip_loss"] = clip_loss
            total_loss = self.clip_weight * clip_loss + sup_loss
        else:
            out_dict = {}
            out_dict["sup_loss"] = sup_loss
            out_dict["clip_loss"] = 0.0
            total_loss = sup_loss
        out_dict["loss"] = total_loss
        return out_dict

     def training_step(self, batch, batch_idx):
        # Labels are provided
        inputs = batch
        assert len(batch) == 2
        if len(batch) == 2:
            inputs = batch[0]
        outputs = self.forward(*inputs)
        out_dict = self.compute_loss(outputs, batch)
        self.log_dict(out_dict, on_epoch=True, sync_dist=True)
        return out_dict["loss"]
     
     def validation_step(self, batch, batch_idx):
        # Labels are provided
        inputs = batch
        if len(batch) == 2:
            inputs = batch[0]
        outputs = self.forward(*inputs)
        out_dict = self.compute_loss(outputs, batch)
        val_loss = out_dict['loss']
        self.log_dict({"val_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

     def test_step(self, batch, batch_idx):
        inputs = batch
        if len(batch) == 2:
            inputs = batch[0]
        outputs = self.forward(*inputs)
        out_dict = self.compute_loss(outputs, batch)
        test_loss = out_dict['loss']
        self.log_dict({"test_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True)
        return test_loss