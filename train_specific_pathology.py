import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Lambda

from torchvision import models
from train_dataset import TrainDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
from models.DenseNet import DenseNet121
from torchvision.transforms import ColorJitter
from torchvision.transforms import RandomHorizontalFlip, RandomRotation

# assume we have cleaned data

def to_rgb(image):
    return image.convert('RGB')

# Add the custom transformation to your transformations
transform = transforms.Compose([
    Resize((224, 224)),  # Resize to 224x224 pixels
    RandomHorizontalFlip(),  # Randomly flip the image horizontally
    RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
    Lambda(to_rgb),  # Convert to RGB
    ToTensor(),
    Lambda(lambda x: x.float()),
])

# make dataset
dataset = TrainDataset(csv_file='data/newtrain2023.csv', root_dir='data', specific_idx=8, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
val_size = len(dataset) - train_size  # 20% of the dataset for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

current_dir = os.getcwd()
checkpoint_dir = os.path.join(current_dir, 'checkpoints')
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='DenseNet121-{epoch:02d}-{val_loss:.2f}')

# Load a pretrained DenseNet121 model
model = DenseNet121()
trainer = pl.Trainer(max_epochs=30, callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
