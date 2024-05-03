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
parser.add_argument("--path", type=str, required=True, help="Path to the test data folder")
parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint name")
args = parser.parse_args()

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

test_dataset = TestDataset(root_dir=args.path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = xrv.models.DenseNet(weights=args.checkpoint)

pathologies = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
model_output_pathologies = {
    'Atelectasis': [0],
    'Consolidation': [1],
    '': [2, 5, 6, 10, 11, 12, 13],
    'Pneumothorax': [3],
    'Edema': [4],
    'Pleural Effusion': [7],
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

def sort_predictions(avg_pred, index_list):
    sorted_pred = [0] * 9
    for i in range(len(index_list)):
        if index_list[i] == 0:
            sorted_pred[i] = 0.0
        else:
            # scale values to -1,1
            sorted_pred[i] = (2*avg_pred[index_list[i]])-1
    return sorted_pred

pids = []
predictions = []

with torch.no_grad():
    for pid, patient_images in test_loader:
        patient_outputs = []
        for image in patient_images:
            output = model(image)
            # raw score for each of the 9 labels
            scores = output.squeeze().tolist()  # remove batch dimension
            patient_outputs.append(scores)

        # average predictions for each patient
        avg_prediction = [sum(x) / len(x) for x in zip(*patient_outputs)]

        pids.append(pid)
        predictions.append(sort_predictions(avg_prediction, index_list))
        print(predictions[-1])

# save sorted predictions in csv
sorted_predictions = sorted(zip(pids, predictions))
labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 'Pleural Effusion',
          'Pleural Other', 'Fracture', 'Support Devices']

with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id'] + labels)
    for pid, prediction in sorted_predictions:
        writer.writerow([pid.item()] + prediction)
