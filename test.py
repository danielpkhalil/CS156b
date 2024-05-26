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

def to_rgb(image):
    return image.convert('RGB')

# Add the custom transformation to your transformations
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    Resize((224, 224)),
    Lambda(to_rgb),
    ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_dataset = TestDataset(root_dir='data/train', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

checkpoint_path = 'checkpoints/' + 'CNN-epoch=09-val_loss=0.00.ckpt'
model = CNNModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
model.eval()

pids = []
predictions = []

with torch.no_grad():
    for pid, patient_images in test_loader:
        patient_outputs = []
        for image in patient_images:
            image = image.cuda()
            output = model(image)
            # raw score for each of the 9 labels
            scores = output.squeeze().tolist()  # remove batch dimension
            patient_outputs.append(scores)

        # average predictions for each patient
        avg_prediction = [sum(x) / len(x) for x in zip(*patient_outputs)]

        pids.append(pid)
        predictions.append(avg_prediction)

# save sorted predictions in csv
sorted_predictions = sorted(zip(pids, predictions))
labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion',
          'Pleural Other', 'Fracture', 'Support Devices']

with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id'] + labels)
    for pid, prediction in sorted_predictions:
        writer.writerow([pid.item()] + prediction)
