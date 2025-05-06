import os
import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import AutoFHRModel
from src.models.losses import introduced_loss

class TestModel(unittest.TestCase):
    """Test cases for the model"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(2025)
        tf.random.set_seed(2025)
        
        # Create a model builder
        self.model_builder = AutoFHRModel(random_seed=2025)
        
        # Sample input shape
        self.input_shape = (1000, 40) 
        
        # Sample batch of data
        self.sample_data = np.random.random((2, *self.input_shape))
        
    def test_model_creation(self):
        """Test that the model can be created with the specified parameters"""
        model = self.model_builder.build(
            input_shape=self.input_shape,
            filters=40,
            kernel_size=20,
            dilation_rates=[2**i for i in range(8)]
        )
        
        # Check that model is a valid Keras model
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape[1:], self.input_shape)
        
        # Check output shape (should match time dimension of input with 1 channel)
        self.assertEqual(model.output_shape[1:], (self.input_shape[0], 1))
    
    def test_model_compilation(self):
        """Test that the model can be compiled"""
        model = self.model_builder.build(
            input_shape=self.input_shape,
            filters=40,
            kernel_size=20,
            dilation_rates=[2**i for i in range(8)]
        )
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss=introduced_loss,
            metrics=['accuracy']
        )
        
        # Check that the model has been compiled
        self.assertTrue(model._is_compiled)
    
    def test_model_prediction(self):
        """Test that the model can make predictions"""
        model = self.model_builder.build(
            input_shape=self.input_shape,
            filters=40,
            kernel_size=20,
            dilation_rates=[2**i for i in range(8)]
        )
        
        # Make a prediction
        predictions = model.predict(self.sample_data)
        
        # Check prediction shape
        expected_shape = (self.sample_data.shape[0], self.input_shape[0], 1)
        self.assertEqual(predictions.shape, expected_shape)
        
        # Check that predictions are in the valid range for sigmoid output [0, 1]
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_loss_function(self):
        """Test the custom loss function"""
        y_true = np.zeros((2, 100, 1))
        y_true[0, 20:40, 0] = 1.0
        y_true[0, 60:80, 0] = 1.0
        y_true[1, 30:50, 0] = 1.0
        y_true[1, 70:90, 0] = 1.0
        
        y_pred = np.random.random((2, 100, 1))
        
        # Calculate loss
        loss = introduced_loss(y_true, y_pred)
        self.assertTrue(np.isscalar(loss.numpy()))
        self.assertTrue(loss.numpy() > 0)

if __name__ == '__main__':
    unittest.main() 