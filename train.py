from models.CNN import CNNModel
from train_dataset import TrainDataset

import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Lambda

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# assume we have cleaned data

# transforms defined here
transform = transforms.Compose([
    Resize((224, 224)),  # Resize to 224x224 pixels
    ToTensor(),
    Lambda(lambda x: x.float()),
])

# make dataset
dataset = TrainDataset(csv_file='data/newtrain2023.csv', root_dir='data', transform=transform)

# use custom dataset for dataloader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

current_dir = os.getcwd()
checkpoint_dir = os.path.join(current_dir, 'checkpoints')
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='CNN-{epoch:02d}-{val_loss:.2f}')

model = CNNModel()
trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloaders=data_loader)
