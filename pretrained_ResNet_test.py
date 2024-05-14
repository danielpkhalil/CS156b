from pretrained_test_dataset import TestDataset

import os
import csv
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose, Lambda

import torchxrayvision as xrv
import skimage

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, required=True, help="Path to the test csv")
parser.add_argument("--data_path", type=str, required=True, help="Path to the test data folder")
parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint name")
args = parser.parse_args()

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])

test_dataset = TestDataset(csv_file=args.csv_path, root_dir=args.data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = xrv.models.ResNet(weights=args.checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pathologies = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
model_output_pathologies = {
    'Atelectasis': [0],
    'Consolidation': [1],
    '': [2, 5, 6, 10, 11, 12, 13],
    'Pneumothorax': [3],
    'Edema': [4],
    'Pleural Effusion': [7],
    'Pleural Other': [9],
    'Pneumonia': [8],
    'Cardiomegaly': [10],
    'Lung Lesion': [14],
    'Fracture': [15],
    'Lung Opacity': [16],
    'Enlarged Cardiomediastinum': [17]
}

index_list = []
for i, p in enumerate(pathologies):
  if p in model_output_pathologies:
    index_list.append(model_output_pathologies[p][0])
  else:
    index_list.append(0)

def sort_predictions(pred, index_list):
    sorted_pred = [0] * 9
    for i in range(len(index_list)):
        if index_list[i] == 0:
            sorted_pred[i] = 0.0
        else:
            # scale values to -1,1
            sorted_pred[i] = (2*pred[index_list[i]])-1
    return sorted_pred

ids = []
predictions = []

with torch.no_grad():
    for id, image in test_loader:
        output = model(image.to(device))
        # raw score for each of the 9 labels
        prediction = output.squeeze().tolist()  # remove batch dimension
        ids.append(id)
        predictions.append(sort_predictions(prediction, index_list))

# save sorted predictions in csv
sorted_predictions = sorted(zip(ids, predictions))
labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion',
          'Pleural Other', 'Fracture', 'Support Devices']

with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id'] + labels)
    for id, prediction in sorted_predictions:
        writer.writerow([id.item()] + prediction)
