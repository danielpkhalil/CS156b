import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Lambda
from torch.utils.data import Dataset, DataLoader
import torchxrayvision as xrv
import skimage
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    # def __getitem__(self, index):
    #     img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
    #
    #     #image = Image.open(img_path)
    #     image = skimage.io.imread(img_path, as_gray=False)
    #     image = xrv.datasets.normalize(image, 255)
    #     if len(image.shape) == 3:
    #         image = image.mean(2)
    #     image = image[None, ...]
    #
    #     if self.transform:
    #         image = self.transform(image)
    #
    #     id = int(self.annotations.iloc[index, 0])
    #     return id, image

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])


        #image = Image.open(img_path)
        image = xrv.utils.load_image(img_path)

        if self.transform:
            image = self.transform(image)

        id = int(self.annotations.iloc[index, 0])
        return id, image
