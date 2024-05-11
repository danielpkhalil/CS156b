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

# model_urls['resnet50-res512-all'] = {
#     "description": 'This model was trained on the datasets pc-nih-rsna-siim-vin at a 512x512 resolution.',
#     "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt',
#     "labels": ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
#                'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
#                'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 
#                'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'],
#     "op_threshs": [0.51570356, 0.50444704, 0.53787947, 0.50723547, 0.5025118, 0.5035252, 0.5038076, 0.51862943, 0.5078151, 0.50724894, 0.5056339, 0.510706, 0.5053923, 0.5020846, np.nan, 0.5080557, 0.5138526, np.nan],
#     "ppv80_thres": [0.690908, 0.720028, 0.7303882, 0.7235838, 0.6787441, 0.7304924, 0.73105824, 0.6839408, 0.7241559, 0.7219969, 0.6346738, 0.72764945, 0.7285066, 0.5735704, np.nan, 0.69684714, 0.7135549, np.nan]
# }

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
        output = model(image)
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
