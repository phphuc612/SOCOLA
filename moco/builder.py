# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOCOLA(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, sub_batch_size, dim=128, mlp_dim=4096, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SOCOLA, self).__init__()

        self.sub_batch_size = sub_batch_size
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.T = nn.Parameter(
            torch.ones([]) * T
        )
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_img = base_encoder(num_classes=mlp_dim)
        print(self.encoder_img)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True): 
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

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
        
            shuffled_idx = torch.randperm(all_query_imgs.shape[0])
            key_img_embeds = key_img_embeds[shuffled_idx]

            all_labels = torch.empty_like(shuffled_idx)
            all_labels.scatter_(0, shuffled_idx, torch.arange(
                len(shuffled_idx)
            ))
            all_labels = all_labels.to(device)
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
            loss = F.cross_entropy(
                sim, all_labels[start_id:end_id], reduction="mean")
            loss /= len(sub_query_images)
            avg_loss += loss.item()
            if loss.grad_fn is not None:
                loss.backward()
        all_sims = torch.concat(all_sims)

        return avg_loss, all_sims, all_labels

class SOCOLA_Resnet(SOCOLA):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_img.fc.weight.shape[1]
        del self.encoder_img.fc # remove original fc layer

        # projectors
        self.encoder_img.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

class SOCOLA_ViT(SOCOLA):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_img.heads.head.weight.shape[1]
        del self.encoder_img.heads# remove original fc layer

        # projectors
        self.encoder_img.heads = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)
