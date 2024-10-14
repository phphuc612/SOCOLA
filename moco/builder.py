# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, sub_batch_size, dim=128, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.sub_batch_size = sub_batch_size
        self.dim = dim
        self.mlp = mlp
        self.T = nn.Parameter(
            torch.ones([]) * T
        )
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_img = base_encoder(num_classes=dim)
        if mlp:
            dim_mlp = self.encoder_img.fc.weight.shape[1]
            self.encoder_img.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_img.fc
            )
        
    def forward(self, all_query_imgs, all_key_imgs, device):
        """
        Input:
            all_query_imgs: a batch of query images
            all_key_imgs: a batch of key images
        Output:
            loss, logits, labels
        """
        sub_query_images = torch.split(all_query_imgs, self.sub_batch_size)
        sub_key_images = torch.split(all_key_imgs, self.sub_batch_size)
        with torch.no_grad():  # no gradient to keys
            self.T.clamp_(0.001, 1)
            key_img_embeds = []
            for key_imgs in sub_key_images:
                key_img_embeds.append(self.encoder_img(key_imgs))  # keys: NxC
            key_img_embeds = torch.concat(key_img_embeds)
            key_img_embeds = F.normalize(key_img_embeds, dim=1)

            all_labels = torch.arange(all_query_imgs.shape[0], device=device)
        all_sims = []
        avg_loss = 0
        for sub_id, query_imgs in enumerate(sub_query_images):
            start_id = sub_id * self.sub_batch_size
            end_id = start_id + query_imgs.shape[0]

            query_img_embeds = self.encoder_img(query_imgs)
            query_img_embeds = F.normalize(query_img_embeds, dim=1)
            # breakpoint()
            # compute similarities
            sim = query_img_embeds @ key_img_embeds.T / self.T  # NxC * CxN
            all_sims.append(sim.detach())

            # losses
            loss = F.cross_entropy(sim, all_labels[start_id:end_id], reduction="mean")
            loss /= len(sub_query_images)
            avg_loss += loss.item()
            if loss.grad_fn is not None:
                loss.backward()
        all_sims = torch.concat(all_sims)
        
        return avg_loss, all_sims, all_labels

# N x K
