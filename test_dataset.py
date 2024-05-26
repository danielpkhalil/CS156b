import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Lambda
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        id = int(self.annotations.iloc[index, 0])

        try:
            # Attempt to open the image
            image = Image.open(img_path)
        except (FileNotFoundError, OSError) as e:
            # If the image path is not found, handle the error gracefully
            print(f"Error opening image: {img_path}")
            print(e)
            # You can return a default image or any other handling mechanism
            return None, None

        if self.transform:
            image = self.transform(image)
            
        return id, image
