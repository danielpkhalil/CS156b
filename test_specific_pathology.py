import os
import csv
import json

import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Lambda

from torchvision import models
from test_dataset import TestDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
from models.DenseNet import DenseNet121
from torchvision.transforms import ColorJitter
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
import argparse

def to_rgb(image):
    return image.convert('RGB')

# Add the custom transformation to your transformations
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, required=True, help="Path to the test csv")
parser.add_argument("--data_path", type=str, required=True, help="Path to the test data folder")
parser.add_argument("--pathogen_idx", type=int, required=True, help="Index of the pathogen to be trained")
parser.add_argument("--checkpoint", type=str, required=False, help="Model checkpoint name")
args = parser.parse_args()

transform = transforms.Compose([
    Resize((224, 224)),
    Lambda(to_rgb),
    ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_dataset = TestDataset(csv_file=args.csv_path, root_dir=args.data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

checkpoint = torch.load(args.checkpoint)
model = DenseNet121()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

ids = []
predictions = []

with torch.no_grad():
    for id, image in test_loader:
        image = image.cuda()
        output = model(image)
        # raw score
        score = output.squeeze().tolist()  # remove batch dimension
        predictions.append(score)
        ids.append(id)

# save sorted predictions in csv
labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion',
          'Pleural Other', 'Fracture', 'Support Devices']

# Create a dictionary mapping each id to its corresponding prediction
predictions_dict = {id: prediction for id, prediction in zip(ids, predictions)}

# Save the dictionary as a JSON file
filename = f"predictions_{labels[args.pathogen_idx]}.json"
with open(filename, 'w') as f:
    json.dump(predictions_dict, f)
