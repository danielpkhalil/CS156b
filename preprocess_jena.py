from train_dataset import TrainDataset
import numpy as np
from PIL import Image
import cv2

train_dataset = TrainDataset(csv_file='data/newtrain2023.csv', root_dir='data', transform=None)

def preprocess_image(image):
    image_np = np.array(image)

    # 1) min-max scaling -- pixel values between [0, 1] rathen than up to 255
    image_np_normalized = image_np / 255.0

    # 2) resize images to be 256 x 256 pixels (originally 2000+ x 2000+)
    image_normalized = Image.fromarray(np.uint8(image_np_normalized * 255))
    image_resized = image_normalized.resize((256, 256), Image.LANCZOS)

    # 3) use opencv to crop images
    image_grayscale = np.array(image_resized.convert('L')) # convert to grayscale
    # apply gaussian blur -- smoothes the image and reduces noise from "fake" edges
    image_blurred = cv2.GaussianBlur(image_grayscale, (5, 5), 0)
    # convert grayscale image to black and white
    _, image_bw = cv2.threshold(image_blurred, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # identify contours -- edges present in the black and white image
    contours, _ = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        lung_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(lung_contour)
        # use the lung contour to crop the image to focus on the outline of the lung
        region = image_grayscale[y:y+h, x:x+w]
        region_display = Image.fromarray(region)
        if region_display.mode != 'RGB':
            region_display = region_display.convert('RGB')
        return region_display
    else:
        return image_resized 


for i in range(len(train_dataset)):
    image, y_label = train_dataset.__getitem__(i)
    processed_image = preprocess_image(image)
    #processed_image.show()
    # TODO: how do we want to store the preprocessed images?
