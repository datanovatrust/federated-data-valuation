# src/models/resnet_model.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetModel, self).__init__()
        # Use the updated way to load pretrained weights
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the first convolutional layer to accept 1-channel input
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # Initialize weights of the new conv1 layer
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
