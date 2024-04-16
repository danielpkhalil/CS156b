import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Lambda
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patient_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, index):
        patient_dir = self.patient_dirs[index]
        patient_path = os.path.join(self.root_dir, patient_dir)

        image_paths = []
        for dirpath, _, filenames in os.walk(patient_path):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    image_paths.append(os.path.join(dirpath, filename))

        images = []
        for img_path in image_paths:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        pid = int(patient_dir[3:])
        return pid, images