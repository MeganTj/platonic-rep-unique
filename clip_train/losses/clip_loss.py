import torch
import torch.nn as nn
import torch.nn.functional as F
from clip_train.utils import get_rank, all_gather_batch


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_loss': loss, 'clip_acc': acc, 'logit_scale': logit_scale}

class BCELossWithMasking(nn.Module):
    def __init__(self, pos_weight=None):
        super(BCELossWithMasking, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)  # We will reduce manually

    def forward(self, predictions, labels):
        # Mask the unlabeled examples (where label is -1)
        labels = labels.unsqueeze(1)
        mask = (labels != -1).float()

        # Compute BCELoss for all examples (without reducing)
        loss = self.bce_loss(predictions, labels.float())

        # Apply the mask to ignore unlabeled examples
        loss = loss * mask

        # Return the average loss over the labeled examples
        return loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0).to(predictions.device)