# src/models/image_classifier.py

import logging
from .vit_model import ViTModel
from .resnet_model import ResNetModel

# Configure logging
logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class to instantiate models dynamically.
    """

    @staticmethod
    def create_model(model_name, num_classes):
        model_name = model_name.lower()
        if model_name == 'vit':
            return ViTModel(num_classes=num_classes)
        elif model_name == 'resnet':
            return ResNetModel(num_classes=num_classes)
        else:
            logger.error(f"Model '{model_name}' not supported.")
            raise ValueError("Model not supported")
