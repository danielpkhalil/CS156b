# returns a dataloader of the chexpert data. create variants as needed
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import torchvision
import PIL
import random
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from tqdm import tqdm
import torchvision.models as models
import argparse

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Train a model on the CheXpert dataset.')

# Adding arguments
parser.add_argument('--category_idx', type=int, default=0, help='which pathology index (0-indexed) to train the model on')
parser.add_argument('--mode', type=str, default='lateral', choices=['lateral', 'frontal'], help='the view (either lateral or frontal)')
parser.add_argument('--num_training_points', type=int, default=60000, help='number of images to train the model on. set 0 for entire dataset')
parser.add_argument('--epochs', type=int, default=51, help='number of epochs to train model on')
parser.add_argument('--val_portion', type=float, default=0.075, help='proportion of the training points to divert for validation')
parser.add_argument('--test_cutoff', type=int, default=0, help='number of test dataset to predict on. set 0 for entire dataset')
parser.add_argument('--test_period', type=int, default=10, help='how often (the period, not frequency) that inference should be applied to test')
parser.add_argument('--rotate_dataloaders', type=bool, default=False, help='if you want to train on the entire dataset by rotating dataloaders')
parser.add_argument('--rotate_dataloaders_period', type=int, default=2, help='how often you want to refresh the dataloader (how many epochs per refresh)')

# Parsing arguments
args = parser.parse_args()

# Assigning parsed arguments to variables
CATEGORY_IDX = args.category_idx
MODE = args.mode
NUM_TRAINING_POINTS = args.num_training_points
EPOCHS = args.epochs
VAL_PORTION = args.val_portion
TEST_CUTOFF = args.test_cutoff
TEST_PERIOD = args.test_period
ROTATE_DATALODERS = args.rotate_dataloaders
ROTATE_DATALODERS_PERIOD = args.rotate_dataloaders_period


# ensuring valid parameters
if TEST_PERIOD >= EPOCHS:
    TEST_PERIOD = EPOCHS - 1
assert (MODE in ["lateral", "frontal"])
if NUM_TRAINING_POINTS == 0:
    NUM_TRAINING_POINTS = 99999999999
if TEST_CUTOFF == 0:
    TEST_CUTOFF = 99999999999


# the structure of this dataset will follow CIFAR10 where __getitem__ returns a tuple of (image, target) where target is a list of the class labels
class chexpert_dataset(Dataset):
    def __init__(self, paths, pathology_idx, convert_to_float=True, root_path="data/", width=2300, height=1500, transform=None):
        """
        Initialize a dataset from chexpert
        
        Args:
        paths (list): List of image paths
        convert_to_float (bool): Setting to True turns all marked classes in the label turned to booleans
        root_path (str): The path to get into the folder 156b/data/
        width (int): width to resize all images to
        height (int): height to resize all images to
        transform (torchvision.transforms): The transformation to be applied to the images

        Returns:
        None
        """
        assert pathology_idx >= 0
        assert pathology_idx <= 8

        self.paths = paths
        self.pathology_idx = pathology_idx
        self.root_path = root_path
        self.width = width
        self.height = height
        self.transform = transform

        # create dict where key is path and value is label (label is a list matching the 9 pathologies)
        self.path_to_label = {}
        
        train_file_path = '/groups/CS156b/data/student_labels/train2023.csv'
        # convert csv file into file path and labels
        with open(train_file_path, mode='r') as file:
            csv_reader = csv.reader(file)

            header = next(csv_reader)

            for line in csv_reader:
                curr_path = line[2]
                labels = line[-9:]
                if convert_to_float:
                    for i in range(len(labels)):
                        if labels[i] != "":
                            labels[i] = float(labels[i])
                        else:
                            labels[i] = 0.0
                self.path_to_label[curr_path] = labels

        
    def __len__(self):
        """
        Returns the number of items in the dataset. Needed for definition of dataset object

        Returns:
        int: number of items in the dataset
        """
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_path, self.paths[idx])
        entire_label = self.path_to_label[self.paths[idx]]
        
        # Set label based on pathology index
        label = [0.0] * len(entire_label)  # Initialize all labels to 0.0
        label = 0.0
        if entire_label[self.pathology_idx] == 1.0:
            # label[self.pathology_idx] = 1.0
            label = 1.0
        elif entire_label[self.pathology_idx] == -1.0:
            # label[self.pathology_idx] = -1.0
            label = -1.0
        elif entire_label[self.pathology_idx] == 0.0:
            label = 0.0
        else:
            print("bruh")
            exit()
        label = [label]
        label = torch.tensor(label)

        # Open image and check dimensions
        image = PIL.Image.open(img_path)
        original_width, original_height = image.size

        # Add padding to make the image square
        max_side = max(original_width, original_height)
        padding_left = (max_side - original_width) // 2
        padding_right = max_side - original_width - padding_left
        padding_top = (max_side - original_height) // 2
        padding_bottom = max_side - original_height - padding_top

        image = transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

        if self.transform is not None:
            image = self.transform(image)
        else:
            # Convert image to tensor
            image = image.to(torch.float32)
        image = image * 255

        return (image, label)


def transform_grayscale_to_rgb(image):
    """Convert a grayscale image to RGB by replicating the single channel across all three RGB channels."""
    return image.repeat(3, 1, 1)  # Repeat the single channel three times.

# 4 dataloaders was recommended by the systems on HPC (will give a warning if more than 4 workers)
def get_data(val_portion=VAL_PORTION, batch_size=32, num_workers=4):
    assert val_portion >= 0
    assert val_portion < 1

    train_file_path = '/groups/CS156b/data/student_labels/train2023.csv'
    all_paths = []

    NUM_CATEGORIES = 9

    # add in values with 0 and 1
    with open(train_file_path, mode='r') as file:
        csv_reader = csv.reader(file)

        header = next(csv_reader)
        
        # index of first of last 9 columns
        offset = len(header) - NUM_CATEGORIES
        headers = header[-NUM_CATEGORIES:]

        for line in csv_reader:
            path = line[2]
            line = line[-9:]
            if line[CATEGORY_IDX] != "" and MODE in path and "train/patient64540/study1/view1_frontal.jpg" not in path: # train/patient64540/study1/view1_frontal.jpg is last data point, and invalid path
                all_paths.append((path, line[CATEGORY_IDX]))
    
    random.shuffle(all_paths)
    
    # do class balancing
    num_neg = 0
    num_pos = 0
    for p in all_paths:
        if p[1] == "1.0":
            num_pos += 1
        if p[1] == "-1.0":
            num_neg += 1

    balanced_paths = []
    
    added_negs = 0
    added_pos = 0
    for p in all_paths:
        if p[1] == "1.0" and added_pos <= min(num_neg, num_pos):
            added_pos += 1
            balanced_paths.append(p[0])
        if p[1] == "-1.0" and added_pos <= min(num_neg, num_pos):
            added_negs += 1
            balanced_paths.append(p[0])

    all_paths = balanced_paths

    # add in values with 0
    with open(train_file_path, mode='r') as file:
        csv_reader = csv.reader(file)

        header = next(csv_reader)
        
        # index of first of last 9 columns
        offset = len(header) - NUM_CATEGORIES
        headers = header[-NUM_CATEGORIES:]

        for line in csv_reader:
            path = line[2]
            line = line[-9:]
            if line[CATEGORY_IDX] == "0.0" and MODE in path:
                all_paths.append(path)
    
    random.shuffle(all_paths)
    
    all_paths = all_paths[:min(len(all_paths), NUM_TRAINING_POINTS)]

    num_val = int(val_portion * len(all_paths))
    val_paths = all_paths[:num_val]
    train_paths = all_paths[num_val:]

    # make sure val_portion still allows for training data
    assert len(train_paths) > 0

    train_dataset = chexpert_dataset(
        paths=train_paths,
        pathology_idx=CATEGORY_IDX,
        convert_to_float=True,
        root_path="/groups/CS156b/data/",
        width=3000,
        height=3000,
        transform=torchvision.transforms.Compose([
                         transforms.RandAugment(),
                         transforms.Resize((400, 400)),
                         transforms.ToTensor(),
                         transform_grayscale_to_rgb
                     ])
    )

    val_dataset = chexpert_dataset(
        paths=val_paths,
        pathology_idx=CATEGORY_IDX,
        convert_to_float=True,
        root_path="/groups/CS156b/data/",
        width=3000,
        height=3000,
        transform=torchvision.transforms.Compose([
                        transforms.Resize((400, 400)),
                        transforms.ToTensor(),
                        transform_grayscale_to_rgb
                     ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False
    )

    return train_loader, val_loader

class ResNet50_Modified(nn.Module):
    def __init__(self):
        super(ResNet50_Modified, self).__init__()
        # Load a pretrained DenseNet-169 model
        self.densenet = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        
        # Replace the classifier layer of DenseNet-169
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

        self.densenet = nn.DataParallel(self.densenet)
        
    def forward(self, x):
        return self.densenet(x)



def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    train_loader, val_loader = get_data()
    model = ResNet50_Modified().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    training_loss = []

    chunk_idx = 0

    for epoch in range(EPOCHS):  # run for 5 epochs
        if epoch != 0 and epoch % ROTATE_DATALODERS_PERIOD == 0:
            train_loader, val_loader = get_data()
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            # Optionally clear the cache after an iteration
            # torch.cuda.empty_cache()

        print(f"Finished training for epoch {epoch + 1}.")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels.float()).item()
                # Optionally clear the cache after processing a validation batch
                # torch.cuda.empty_cache()

        print(f'Epoch {epoch+1}, Loss: {sum(training_loss) / len(training_loss)}, Validation Loss: {val_loss / len(val_loader)}')

        if epoch % TEST_PERIOD == 0 and epoch != 0:
    
            # test it the outputs on the test data
            test_csv_path = "data/student_labels/test_ids.csv"
            paths = []
            with open(test_csv_path, mode='r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)

                for line in csv_reader:
                    if MODE in line[1]:
                        paths.append(line[1])
            
            transform=torchvision.transforms.Compose([
                                transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transform_grayscale_to_rgb
                            ])
            
            paths = paths[:min(len(paths), TEST_CUTOFF)]
            model.eval()
            out = {}
            for i in tqdm(range(len(paths))):
                p = paths[i]
                img_path = os.path.join("data/", p)

                # Open image and check dimensions
                image = PIL.Image.open(img_path)
                original_width, original_height = image.size

                max_side = max(original_width, original_height)
                padding_left = (max_side - original_width) // 2
                padding_right = max_side - original_width - padding_left
                padding_top = (max_side - original_height) // 2
                padding_bottom = max_side - original_height - padding_top

                image = transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill=0)
                
                if transform is not None:
                    image = transform(image)
                else:
                    # Convert image to tensor
                    image = image.to(torch.float32)
                
                image = image * 255

                # Add a batch dimension: [C, H, W] -> [1, C, H, W]
                image = image.unsqueeze(0)

                # Move image to the same device as the model
                image = image.to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(image)
                    out[p] = output.item()
                    # print(output.item())
            
            # replace all the old files
            f = open(f"2024/pp/output_labels_{MODE}{CATEGORY_IDX}_dense.json", "r")
            old = json.load(f)
            f.close()

            for key in out:
                old[key] = out[key]

            # save the values
            with open(f"2024/pp/output_labels_{MODE}{CATEGORY_IDX}_dense.json", "w") as outfile:
                json.dump(old, outfile, indent=4)


if __name__ == "__main__":
    train_model()
