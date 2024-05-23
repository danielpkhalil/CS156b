import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Lambda

from torchvision import models
from train_dataset import TrainDataset
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DenseNet121(pl.LightningModule):
    def __init__(self):
        super(DenseNet121, self).__init__()
        # Load a pretrained DenseNet121 model
        self.base_model = models.densenet121(pretrained=True)

        # Replace the classifier layer to match the number of classes in your dataset
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = torch.nn.Linear(num_ftrs, 1)  # Assuming binary classification

        # Define your loss function here
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Initialize lists to store the outputs of each training and validation step
        self.train_outputs = []
        self.val_outputs = []

    def forward(self, x):
        x = self.base_model(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.unsqueeze(1)
        loss = self.criterion(outputs, targets)
        self.train_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.unsqueeze(1)
        loss = self.criterion(outputs, targets)
        self.val_outputs.append(loss)
        self.log('val_loss', loss)

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_outputs).mean()
        self.log('avg_train_loss', avg_loss)
        self.train_outputs.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_outputs).mean()
        self.log('avg_val_loss', avg_loss)
        self.val_outputs.clear()

    def configure_optimizers(self):
        # Define your optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)  # Change learning rate here
        return optimizer

    # def on_train_end(self):
    #     # Read the logged metrics from the CSV file
    #     metrics = pd.read_csv(f'{self.logger.log_dir}/metrics.csv')
    
    #     # Preprocess the data
    #     epochs = metrics['epoch'].unique()
    #     max_epoch = epochs.max()
    #     train_loss = metrics['avg_train_loss'].dropna().reindex(range(max_epoch + 1), fill_value=np.nan).tolist()
    #     val_loss = metrics['avg_val_loss'].dropna().reindex(range(max_epoch + 1), fill_value=np.nan).tolist()
    
    #     # Create a new DataFrame
    #     data = pd.DataFrame({
    #         'epoch': epochs,
    #         'avg_train_loss': train_loss,
    #         'avg_val_loss': val_loss
    #     })
    
    #     # Create a figure and axes
    #     fig, ax = plt.subplots()
    
    #     # Plot the average training and validation losses
    #     ax.plot(data['epoch'], data['avg_train_loss'], label='Avg Train Loss')
    #     ax.plot(data['epoch'], data['avg_val_loss'], label='Avg Val Loss')
    
    #     # Set the title and labels
    #     ax.set_title('Model Loss Across Epochs')
    #     ax.set_ylabel('Loss')
    #     ax.set_xlabel('Epoch')
    
    #     # Add a legend
    #     ax.legend()
    
    #     # Save the figure to a file and close it
    #     fig.savefig(f'{self.logger.log_dir}/loss_plot.png')
    #     plt.close(fig)
