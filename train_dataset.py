import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor
import os
from sklearn.utils import resample

class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, specific_idx=None, transform=None, balance=True, smoothing=False):
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[~self.annotations['Path'].str.contains('train/patient64540/study1/view1_frontal.jpg')]
        self.root_dir = root_dir
        self.specific_idx = specific_idx
        self.transform = transform
        self.smoothing = smoothing

        # Reduce the annotations to just the label that corresponds to the specific idx
        if self.specific_idx is not None:
            self.annotations = self.annotations[['Path', self.annotations.columns[7 + self.specific_idx]]]

        # Balance the dataset if balance parameter is True
        if balance:
            self.annotations = self.balance_dataset(self.annotations)

    def balance_dataset(self, annotations):
        # Separate classes
        df_class_1 = annotations[annotations.iloc[:, 1] == 1]
        df_class_0 = annotations[annotations.iloc[:, 1] == 0]
        df_class_minus_1 = annotations[annotations.iloc[:, 1] == -1]
    
        # Find the number of samples in the smallest class
        min_samples = min(len(df_class_1), len(df_class_0), len(df_class_minus_1))
    
        # Downsample each class to match the smallest class
        df_class_1_downsampled = resample(df_class_1, replace=False, n_samples=min_samples, random_state=123)
        df_class_0_downsampled = resample(df_class_0, replace=False, n_samples=min_samples, random_state=123)
        df_class_minus_1_downsampled = resample(df_class_minus_1, replace=False, n_samples=min_samples, random_state=123)
    
        # Combine downsampled classes
        df_downsampled = pd.concat([df_class_1_downsampled, df_class_0_downsampled, df_class_minus_1_downsampled])
    
        return df_downsampled

    def smooth_labels(self, y, factor=0.1):
        # if the label smoothing factor is zero, return the original labels
        if factor == 0:
            return y

        # if the label smoothing factor is not zero, smooth the labels
        y_smooth = y * (1 - factor) + (1 - y) * factor
        return y_smooth

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0].replace('CheXpert-v1.0/', ''))

        try:
            # Attempt to open the image
            image = Image.open(img_path)
        except (FileNotFoundError, OSError) as e:
            # If the image path is not found, handle the error gracefully
            print(f"Error opening image: {img_path}")
            print(e)
            # You can return a default image or any other handling mechanism
            return None, None

        image = Image.open(img_path)
        y_label = torch.tensor(self.annotations.iloc[index, 1])
        y_label = y_label.float()

        if self.transform:
            image = self.transform(image)
        if self.smoothing:
            y_label = self.smooth_labels(y_label)

        return (image, y_label)
