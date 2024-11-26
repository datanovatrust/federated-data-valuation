# tests/test_config.py

import unittest
import sys
import os

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attacks.config import RMIAConfig

class TestRMIAConfig(unittest.TestCase):
    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        self.assertEqual(RMIAConfig.GAMMA, 1.0)
        self.assertEqual(RMIAConfig.BETA, 0.5)
        self.assertEqual(RMIAConfig.NUM_REFERENCE_MODELS, 5)
        self.assertEqual(RMIAConfig.NUM_Z_SAMPLES, 1000)
    
    def test_modify_config(self):
        """Test modifying configuration parameters."""
        RMIAConfig.GAMMA = 0.9
        RMIAConfig.BETA = 0.4
        RMIAConfig.NUM_REFERENCE_MODELS = 10
        RMIAConfig.NUM_Z_SAMPLES = 500

        self.assertEqual(RMIAConfig.GAMMA, 0.9)
        self.assertEqual(RMIAConfig.BETA, 0.4)
        self.assertEqual(RMIAConfig.NUM_REFERENCE_MODELS, 10)
        self.assertEqual(RMIAConfig.NUM_Z_SAMPLES, 500)
    
    def test_add_new_hyperparameters(self):
        """Test adding new configuration parameters."""
        RMIAConfig.LEARNING_RATE = 0.01
        self.assertEqual(RMIAConfig.LEARNING_RATE, 0.01)

if __name__ == '__main__':
    unittest.main()
