# -*- coding: utf-8 -*-

# Libraries
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torchvision
from skimage.util import random_noise
import numpy as np

class WCAMClassification(Dataset):
    """
    A class that defines a dataset with 
    images corresponding to WCAMs and
    labels corresponding to the status
    """

    def __init__(self, csv_file, root_dir, transform=None) -> None:
        """
        initializer
        - root_dir: the directory where the data is located
        - csv file : the csv containing the 
        - transform : whethter the images should be transformed or not
        """
        self.labels = csv_file
        self.dir = root_dir
        self.transform = transform

    def __len__(self):        
        return self.labels.shape[0]

    def __getitem__(self, idx):

        img_name = self.labels.iloc[idx]['name']
        label = self.labels.iloc[idx]['label'] 

        img_path = os.path.join(self.dir, img_name)
        img = Image.open(img_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        else:
            t = transforms.Compose([transforms.ToTensor()])
            img = t(img)

        return img, label

class ImageNetDataset(Dataset):
    """
    A class that defines a dataset with 
    images corresponding to WCAMs and
    labels corresponding to the status
    """

    def __init__(self, csv_file, root_dir, transform=None) -> None:
        """
        initializer
        - root_dir: the directory where the data is located
        - csv file : the csv containing the 
        - transform : whethter the images should be transformed or not
        """
        self.labels = csv_file
        self.dir = root_dir
        self.transform = transform

    def __len__(self):        
        return self.labels.shape[0]

    def __getitem__(self, idx):

        img_name = self.labels.iloc[idx]['name']
        label = self.labels.iloc[idx]['label'] 

        img_path = os.path.join(self.dir, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        else:
            t = transforms.Compose([transforms.ToTensor()])
            img = t(img)

        return img, label, img_name