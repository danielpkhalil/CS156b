import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor
import os
from sklearn.utils import resample

class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, specific_idx=None, transform=None, balance=False):
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[~self.annotations['Path'].str.contains('train/patient64540/study1/view1_frontal.jpg')]
        self.root_dir = root_dir
        self.specific_idx = specific_idx
        self.transform = transform

        # Reduce the annotations to just the label that corresponds to the specific idx
        if self.specific_idx is not None:
            self.annotations = self.annotations[['Path', self.annotations.columns[7 + self.specific_idx]]]

        # Balance the dataset if balance parameter is True
        if balance:
            self.annotations = self.balance_dataset()

    def balance_dataset(self):
        # Separate majority and minority classes
        df_majority = self.annotations[self.annotations.iloc[:, 1] == 1]
        df_minority = self.annotations[self.annotations.iloc[:, 1] != 1]

        # Downsample majority class
        df_majority_downsampled = resample(df_majority, 
                                            replace=False,    # sample without replacement
                                            n_samples=len(df_minority),     # to match minority class
                                            random_state=123) # reproducible results

        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])

        return df_downsampled

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

        return (image, y_label)
