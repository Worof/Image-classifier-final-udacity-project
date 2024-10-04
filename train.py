# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MlHjSA_RSnMRVNTbAPZytWBr6PvEIFtp
"""

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

# Argument parsing
parser = argparse.ArgumentParser(description='Train a neural network')
parser.add_argument('data_dir', type=str, help='Directory of training data')
parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or resnet18)')
parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units for the classifier')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args = parser.parse_args()

# Load a pre-trained model dynamically based on architecture
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    in_feat = 25088  # VGG's first linear layer in_features
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
    in_feat = model.fc.in_features  # ResNet's first linear layer in_features
else:
    raise ValueError("Unsupported architecture")

# Freeze parameters of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Define the classifier dynamically based on the architecture
if args.arch.startswith('resnet'):
    model.fc = nn.Sequential(nn.Linear(in_feat, args.hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(args.hidden_units, 102),
                             nn.LogSoftmax(dim=1))
else:
    model.classifier = nn.Sequential(nn.Linear(in_feat, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

# Set up the optimizer for either the `fc` layer (ResNet) or the `classifier` (VGG)
if args.arch.startswith('resnet'):
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Data loading and training steps will go here
# Example: Training the model
# Then proceed to the training loop (like the `train_model()` function).