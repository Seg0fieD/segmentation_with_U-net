# import numpy as np
# import pandas as pd 
# import cv2
# import os 

# folder = 'data/train'
# images = []
# mask = []

# with open(folder) as f:
    
    
import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
data_dir = 'data/train'
mask_dir = 'data/train_masks'

train_img_dir = 'data/train_img_'
val_img_dir = 'data/val_img_'
train_mask_dir = 'data/train_mask_'
val_mask_dir = 'data/val_mask_'

# Create directories if they don't exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

# Get list of all images and masks
image_files = sorted(os.listdir(data_dir))
mask_files = sorted(os.listdir(mask_dir))

# Split the images and masks into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42)

# Copy the files to their respective directories
for img in train_images:
    shutil.copy(os.path.join(data_dir, img), os.path.join(train_img_dir, img))

for img in val_images:
    shutil.copy(os.path.join(data_dir, img), os.path.join(val_img_dir, img))

for mask in train_masks:
    shutil.copy(os.path.join(mask_dir, mask), os.path.join(train_mask_dir, mask))

for mask in val_masks:
    shutil.copy(os.path.join(mask_dir, mask), os.path.join(val_mask_dir, mask))

print("Data split into training and validation sets successfully.")
