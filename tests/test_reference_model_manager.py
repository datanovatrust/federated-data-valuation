# tests/test_reference_model_manager.py

import unittest
import sys
import os
import torch
from unittest.mock import MagicMock

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attacks.reference_model_manager import ReferenceModelManager

class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

class TestReferenceModelManager(unittest.TestCase):
    def setUp(self):
        self.reference_models = [MockModel(), MockModel()]
        self.manager = ReferenceModelManager(self.reference_models)
    
    def test_initialization(self):
        """Test that the manager initializes with the provided reference models."""
        self.assertEqual(len(self.manager.reference_models), 2)
        self.assertEqual(self.manager.reference_models, self.reference_models)
    
    def test_get_reference_models(self):
        """Test retrieving the reference models."""
        models = self.manager.get_reference_models()
        self.assertEqual(models, self.reference_models)
    
    def test_load_pretrained_models(self):
        """Test loading pretrained models."""
        # Mock the _load_model method to return a MockModel instance
        self.manager._load_model = MagicMock(return_value=MockModel())
        model_paths = ['path/to/model1', 'path/to/model2']
        self.manager.load_pretrained_models(model_paths)
        self.assertEqual(len(self.manager.reference_models), 4)  # 2 initial + 2 loaded
        self.assertTrue(all(isinstance(model, MockModel) for model in self.manager.reference_models))
        # Ensure _load_model was called with correct arguments
        self.manager._load_model.assert_any_call('path/to/model1')
        self.manager._load_model.assert_any_call('path/to/model2')
    
    def test_train_reference_models(self):
        """Test training reference models."""
        # Mock the _train_model method to return a MockModel instance
        self.manager._train_model = MagicMock(return_value=MockModel())
        data_loader = MagicMock()
        self.manager.train_reference_models(data_loader, num_models=3)
        self.assertEqual(len(self.manager.reference_models), 5)  # 2 initial + 3 trained
        self.assertTrue(all(isinstance(model, MockModel) for model in self.manager.reference_models))
        # Ensure _train_model was called correct number of times
        self.assertEqual(self.manager._train_model.call_count, 3)
    
    def test_load_model(self):
        """Test the _load_model method."""
        # Since _load_model is a placeholder, we'll mock it to return a MockModel
        self.manager._load_model = MagicMock(return_value=MockModel())
        model = self.manager._load_model('path/to/model')
        self.assertIsInstance(model, MockModel)
        self.manager._load_model.assert_called_once_with('path/to/model')
    
    def test_train_model(self):
        """Test the _train_model method."""
        # Since _train_model is a placeholder, we'll mock it to return a MockModel
        self.manager._train_model = MagicMock(return_value=MockModel())
        data_loader = MagicMock()
        model = self.manager._train_model(data_loader)
        self.assertIsInstance(model, MockModel)
        self.manager._train_model.assert_called_once_with(data_loader)
        
if __name__ == '__main__':
    unittest.main()
