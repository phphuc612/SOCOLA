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
        
        with torch.no_grad():
            torch.clamp(self.T, 0.001, 1)

        # Key encoder
        key_img_embeds = []
        for key_imgs in sub_key_images:
            key_img_embeds.append(self.key_encoder(key_imgs))  # keys: NxC
        key_img_embeds = torch.concat(key_img_embeds)
        key_img_embeds = F.normalize(key_img_embeds, dim=1)

        shuffled_idx = torch.randperm(all_query_imgs.shape[0])
        key_img_embeds = key_img_embeds[shuffled_idx]

        all_labels = torch.empty_like(shuffled_idx)
        all_labels.scatter_(0, shuffled_idx, torch.arange(
            len(shuffled_idx)
        ))
        all_labels = all_labels.to(device)
        # End of key encoder
        
        all_sims = []
        avg_loss = 0
        for sub_id, query_imgs in enumerate(sub_query_images):
            start_id = sub_id * self.sub_batch_size
            end_id = start_id + query_imgs.shape[0]

            query_img_embeds = self.query_encoder(query_imgs)
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
                loss.backward(retain_graph=True)
        all_sims = torch.concat(all_sims)

        return avg_loss, all_sims, all_labels
    
    