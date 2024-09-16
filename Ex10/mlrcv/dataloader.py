import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import json
from torchvision import transforms
import typing


transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_val = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class ImageLoader(Dataset):
    def __init__(self, root_dir, annotation_dict, transform=None):
        self.root_dir = root_dir
        self.annotation_dict = annotation_dict
        self.transform = transform
        self.images_annotation = []
        self.load_images()

    def load_images(self):
        """
        This function loads the dataset images (given the root_dir) and it's annotations,
        saving it to images_annotation list

        Args:

        Returns:
            
        """
        for img_name, label in self.annotation_dict.items():
            img_path = os.path.join(self.root_dir, img_name)
            img = Image.open(img_path)
            img = img.convert('RGB')
            
            
            
            self.images_annotation.append({
                'img': img,
                'label': label

            })
        
        pass

    def __len__(self):
        return len(self.images_annotation)

    def __getitem__(self, index):
        data = self.images_annotation[index]

        if self.transform is not None:
            data['img'] = self.transform(data['img'])

        return data
