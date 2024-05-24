import os

import torch.cuda
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
import argparse

# assume we have cleaned data

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, required=True, help="Path to the test csv")
parser.add_argument("--data_path", type=str, required=True, help="Path to the test data folder")
parser.add_argument("--pathogen_idx", type=int, required=True, help="Index of the pathogen to be trained")
parser.add_argument("--checkpoint", type=str, required=False, help="Model checkpoint name")
args = parser.parse_args()

print(torch.cuda.device_count())

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
#zoom? other augmentations
#mse loss
#AdamW
#efficientnet?
#normalizing

num_epochs = 10
num_workers = 4
#num_devices = 4
batch_size = 64

# make dataset
dataset = TrainDataset(csv_file=args.csv_path, root_dir=args.data_path, specific_idx=args.pathogen_idx, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
val_size = len(dataset) - train_size  # 20% of the dataset for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

current_dir = os.getcwd()
checkpoint_dir = os.path.join(current_dir, 'checkpoints')
#checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename=str(args.pathogen_idx) + 'DenseNet121-{epoch:02d}-{val_loss:.2f}')
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, 
                                      filename=str(args.pathogen_idx) + 'DenseNet121-{epoch:02d}-{val_loss:.2f}',
                                      save_top_k=1, 
                                      monitor='val_loss', 
                                      mode='min', 
                                      save_last=True)

# Load a pretrained DenseNet121 model
model = DenseNet121()

strategy = pl.strategies.DDPStrategy(static_graph = True)

if args.checkpoint is not None:
    trainer = pl.Trainer(devices=[0,1], strategy=strategy, max_epochs=num_epochs, callbacks=[checkpoint_callback], resume_from_checkpoint=args.checkpoint)
else:
    trainer = pl.Trainer(devices=[0,1], strategy=strategy, max_epochs=num_epochs, callbacks=[checkpoint_callback])

print('trainer world size: ', trainer.world_size)
print('num nodes: ', trainer.num_nodes)
print('accelerator: ', trainer.accelerator)
print('ids: ', trainer.device_ids)
print('train num workers: ', train_loader.num_workers)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
