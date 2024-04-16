from models.CNN import CNNModel
from test_dataset import TestDataset

import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose, Lambda

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

transform = Compose([
    Resize((224, 224)),  # should add crop image later
    ToTensor(),
    Lambda(lambda x: x.float()),
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