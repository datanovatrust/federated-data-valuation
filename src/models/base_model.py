# src/models/base_model.py

import torch.nn as nn
import torch.nn.functional as F
import logging
import torch

# Configure logging for models
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class BaseModel(nn.Module):
    """
    Base class for models.
    """

    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict_proba(self, x):
        """
        Computes class probabilities for input x.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities
