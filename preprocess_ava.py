# ava

from train_dataset import TrainDataset

from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Lambda

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


train_dataset = TrainDataset(csv_file='data/newtrain2023.csv', root_dir='data', transform=None)
image, y_label = train_dataset.__getitem__(39)
#image.show()
image_np = np.array(image)

# Adjust the blur to balance detail suppression and feature retention
blurred_image = cv2.blur(image_np, (60, 60))

def box_around_torso(img):
    if len(img.shape) == 2:  # Grayscale
        gray = img
    elif len(img.shape) == 3:  # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format")

    # Edge detection to capture significant 
    for i in[5]:
        for j in [320]:
            print(f"{i},{j}")
            edges = cv2.Canny(gray, i, j)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Focus on the largest contour assuming it's the torso
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle in green


            # Convert to PIL for display
            image_pil = Image.fromarray(img)
            plt.imshow(image_pil)
            plt.axis('off')
            plt.show()

box_around_torso(blurred_image)
