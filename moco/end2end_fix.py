import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class End2End(nn.Module):

    def __init__(self, 
                 base_encoder: nn.Module, 
                 sub_batch_size: int, 
                 dim: int = 128, 
                 T: float = 0.07, 
                 mlp: bool = False):
        super(End2End, self).__init__()

        self.sub_batch_size = sub_batch_size
        self.dim = dim
        self.mlp = mlp
        self.T = nn.Parameter(torch.ones([]) * T)

        self.query_encoder = copy.deepcopy(base_encoder(num_classes=dim))
        self.key_encoder = copy.deepcopy(base_encoder(num_classes=dim))

        if mlp:
            dim_mlp = self.query_encoder.fc.weight.shape[1]
            self.query_encoder.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.query_encoder.fc
            )
            self.key_encoder.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.key_encoder.fc
            )

        self.random_init_()

    def random_init_(self):
        for m in self.query_encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        for m in self.key_encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, all_query_imgs, all_key_imgs):
        """
        Input:
            all_query_imgs: a batch of query images
            all_key_imgs: a batch of key images
        Output:
            loss, logits, labels
        """
        device=all_query_imgs.device
        
        query_feats = self.query_encoder(all_query_imgs)
        key_feats = self.key_encoder(all_key_imgs)

        query_feats = F.normalize(query_feats, dim=1)
        key_feats = F.normalize(key_feats, dim=1)

        sim = query_feats @ key_feats.T / self.T
        all_sims = sim.detach()

        all_labels = torch.arange(all_query_imgs.shape[0], device=device)

        return all_sims, all_labels
    
    