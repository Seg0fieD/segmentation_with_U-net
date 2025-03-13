# Carvana Image Segmentation with U-Net

This repository contains a PyTorch implementation of a U-Net architecture for vehicle segmentation using the Carvana Image Masking Challenge dataset.

## Project Overview

The Carvana Image Masking Challenge involves automatically identifying the boundaries of cars in images. This project uses a U-Net architecture to generate high-quality segmentation masks that separate vehicles from their backgrounds.

## Dataset

The dataset is from the [Carvana Image Masking Challenge on Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge). It contains pairs of car images and their corresponding segmentation masks.

## Repository Structure

```
├── data/
│   ├── train/              # Training images
│   ├── train_masks/        # Training masks
│   ├── processed/          # Processed and split data
│       ├── train_img_/     # Training split images
│       ├── train_mask_/    # Training split masks
│       ├── val_img_/       # Validation split images
│       ├── val_mask_/      # Validation split masks
├── outputs/                # Model predictions
├── data_preprocess.py      # Data preprocessing script
├── dataset.py              # Dataset loading utilities
├── model_unet.py           # U-Net model implementation
├── train.py                # Training script
├── utils.py                # Utility functions
└── README.md               # Project documentation
```



## Usage

### Data Preprocessing

Split the dataset into training and validation sets:

```bash
python data_preprocess.py
```

### Training

Train the U-Net model:

```bash
python train.py
```

Hyperparameter can be modified in  `train.py`: such as learning rate, epochs, batch size, image height and width.


### Model Architecture

The U-Net architecture implemented in this project consists of:
- Encoder path (contracting): Series of double convolution blocks followed by max pooling
- Bottleneck: Double convolution at the bottom
- Decoder path (expanding): Series of up-convolutions and concatenations with skip connections
- Final 1x1 convolution to map to output segmentation

## Evaluation

Model performance is evaluated using:
- Pixel-wise accuracy
- Dice coefficient (F1 score)

## Results

The model outputs predicted masks in the `outputs/` directory. Each prediction includes a timestamp for tracking experiments.

## Dependencies

PyTorch, torchvision, albumentations, OpenCV, scikit-image, PIL, tqdm

## Acknowledgements

- [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)
- U-Net: [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
