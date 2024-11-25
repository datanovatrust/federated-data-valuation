# src/models/vit_model.py

from transformers import ViTForImageClassification
from .base_model import BaseModel
import logging
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)

class ViTModel(BaseModel):
    """
    Vision Transformer (ViT) model for image classification.
    """

    def __init__(self, num_classes=10):
        super(ViTModel, self).__init__(num_classes)
        try:
            self.model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                num_labels=num_classes
            )
            logger.info(f"Loaded pre-trained ViT model with {num_classes} labels.")
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            raise

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits  # ViTForImageClassification returns an object with logits
        return logits
