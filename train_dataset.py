import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor
import os

class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, specific_idx=None, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.specific_idx = specific_idx
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        # Check if the image path starts with "CheXpert-v1.0/"
        if img_path.startswith("CheXpert-v1.0/"):
            # If it does, remove the "CheXpert-v1.0/" prefix
            img_path = img_path[len("CheXpert-v1.0/"):]

        #image = read_image(img_path)
        image = Image.open(img_path)
        y_label = torch.tensor(self.annotations.iloc[index, 7:])
        y_label = y_label.float()
        if self.specific_idx != None:
            y_label = y_label[self.specific_idx]

        if self.transform:
            image = self.transform(image)

        return (image, y_label)