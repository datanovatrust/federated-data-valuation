# src/models/model.py

import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision import models
import logging

# Configure logging for model
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers if they don't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ImageClassifier(nn.Module):
    def __init__(self, model_name='vit', num_classes=10):
        super(ImageClassifier, self).__init__()
        self.model_name = model_name
        try:
            if model_name == 'vit':
                self.model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224-in21k',
                    num_labels=num_classes
                )
                logger.info(f"Loaded pre-trained ViT model with {num_classes} labels.")
            elif model_name == 'resnet':
                self.model = models.resnet18(pretrained=True)
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                logger.info(f"Loaded pre-trained ResNet18 model with {num_classes} labels.")
            else:
                logger.error(f"Model '{model_name}' not supported.")
                raise ValueError("Model not supported")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    
    def forward(self, x):
        return self.model(x)
