import os
from torch.utils.data import Dataset
from PIL import Image
import json

class COCODataset(Dataset):
    def __init__(self, data_dir, transform=None, annotation_dir=None, split="train"):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.annotations = json.load(open(os.path.join(annotation_dir, f"captions_{split}2017.json"), "r"))["images"]
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index) -> tuple:
        img_path = self.annotations[index]["file_name"]
        img_path = os.path.join(self.data_dir, f"{self.split}2017" , img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            query, key = self.transform(img)
        return query, key
    