"""
Mock TensorFlow model implementation for CI testing.

This module provides lightweight mockups of TensorFlow models and functionality
to allow testing without requiring the full TensorFlow installation.
"""

import os
import logging
import json
import numpy as np # type: ignore

logger = logging.getLogger(__name__)

class MockTensorFlowModel:
    """A mock implementation of a TensorFlow model for CI testing."""
    
    def __init__(self, name="mock_model"):
        """Initialize the mock model."""
        self.name = name
        self.is_trained = False
        logger.info(f"Initialized mock TensorFlow model: {name}")
    
    def fit(self, x, y, epochs=10, batch_size=32, validation_split=0.2):
        """Mock training method."""
        logger.info(f"Mock training model {self.name} with {len(x)} samples")
        self.is_trained = True
        # Return mock history
        return {
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
            "val_accuracy": [0.55, 0.65, 0.75, 0.8, 0.85],
            "loss": [0.8, 0.6, 0.4, 0.3, 0.2],
            "val_loss": [0.9, 0.7, 0.5, 0.4, 0.3]
        }
    
    def predict(self, x):
        """Mock prediction method."""
        logger.info(f"Making mock predictions with model {self.name}")
        # Return random predictions
        return np.random.random(size=(len(x), 1))
    
    def evaluate(self, x, y):
        """Mock evaluation method."""
        logger.info(f"Evaluating mock model {self.name}")
        # Return mock metrics
        return [0.2, 0.85]  # loss, accuracy
    
    def save(self, filepath):
        """Mock save method."""
        logger.info(f"Saving mock model to {filepath}")
        if not os.path.exists(os.path.dirname(filepath)):
            logger.warning(f"Directory {os.path.dirname(filepath)} does not exist or is not writable, skipping model save")
            return False
        try:
            # Save a simple JSON file instead of a real model
            with open(f"{filepath}.json", "w") as f:
                json.dump({
                    "model_type": "mock",
                    "name": self.name,
                    "is_trained": self.is_trained
                }, f)
            return True
        except Exception as e:
            logger.warning(f"Error saving mock model: {e}")
            return False
    
    @classmethod
    def load(cls, filepath):
        """Mock load method."""
        logger.info(f"Loading mock model from {filepath}")
        model = cls()
        # Try to load the JSON file if it exists
        json_path = f"{filepath}.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                model.name = data.get("name", "loaded_mock_model")
                model.is_trained = data.get("is_trained", False)
        return model


def create_mock_model(model_type="sequential"):
    """Create a mock TensorFlow model of the specified type."""
    logger.info(f"Creating mock {model_type} model")
    return MockTensorFlowModel(name=f"mock_{model_type}")


# Mock TensorFlow imports for CI environment
def mock_keras_sequential():
    """Return a mock Sequential model."""
    return MockTensorFlowModel(name="mock_sequential")


def mock_keras_model():
    """Return a mock Keras Model."""
    return MockTensorFlowModel(name="mock_functional") 