# src/models/resnet_model.py

import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
import logging
from .base_model import BaseModel
from peft import get_peft_model, LoraConfig  # Import PEFT

# Configure logging
logger = logging.getLogger(__name__)

class ResNetModel(BaseModel):
    def __init__(self, num_classes=10, lora_rank=4):
        super(ResNetModel, self).__init__(num_classes)
        self.lora_rank = lora_rank
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1 for MNIST
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Replace the fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Wrap the model with PEFT's LoRA
        peft_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=["fc"],  # The layers to apply LoRA
            lora_dropout=0.1,
            bias="none",
            task_type="CLASSIFICATION"
        )
        self.model = get_peft_model(self.model, peft_config)

    def forward(self, x):
        return self.model(x)
