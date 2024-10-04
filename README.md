# Image Classifier Project

## Overview
This project was completed as part of the **AWS AI & ML Scholarship Program** in collaboration with **Udacity**. The goal of this project was to build an **Image Classifier** using a deep neural network trained on a dataset of flower images. The classifier predicts the flower species given an input image.

## Project Highlights

- **Framework**: PyTorch
- **Model**: Pre-trained VGG16 model fine-tuned for flower classification
- **Dataset**: 102 flower species dataset
- **Training Environment**: GPU-powered
- **Accuracy**: Achieved over 70% accuracy on the test dataset

## Key Features

- **Train Your Model**: The `train.py` script allows users to train the model on the dataset with flexible parameters.
- **Predict Flower Species**: The `predict.py` script predicts the flower species from an input image using the trained model.
- **Command-line Interface (CLI)**: Both scripts support command-line arguments for customizing the architecture, learning rate, epochs, and more.
- **Save and Load Checkpoints**: Save the trained model and load it later for inference.

## Files

- `train.py`: Script to train the model.
- `predict.py`: Script to make predictions on input images.
- `cat_to_name.json`: JSON file mapping flower categories to their names.

## How to Run

### Training the Model
```bash
python train.py flowers --arch vgg16 --epochs 5 --gpu --save_dir /path/to/save/checkpoint

### Predicting with the Model
```bash
python predict.py /path/to/image /path/to/checkpoint --top_k 5 --category_names cat_to_name.json --gpu

