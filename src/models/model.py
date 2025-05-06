import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import TimeDistributed, MultiHeadAttention

class AutoFHRModel:
    def __init__(self, random_seed=2025):
        """
        Initialize the AutoFHR model 
        
        Parameters:
        -----------
        random_seed: int
            Random seed for weight initialization
        """
        self.weights_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=random_seed)
        tf.random.set_seed(random_seed)
        
    def _conv1d(self, filters, kernel_size, strides=1, padding='same', activation=None, use_bias=True):
        """Helper method to create a Conv1D layer with predefined initialization"""
        return keras.layers.Conv1D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=self.weights_initializer,
            bias_initializer="zeros"
        )
    
    def _causal_conv1d(self, filters, kernel_size, dilation_rate, strides=1, padding='causal', activation=None, use_bias=True):
        """Helper method to create a causal Conv1D layer with dilation"""
        return keras.layers.Conv1D(
            filters=filters, 
            kernel_size=kernel_size, 
            dilation_rate=dilation_rate, 
            strides=strides, 
            padding=padding, 
            activation=activation, 
            use_bias=use_bias, 
            kernel_initializer=self.weights_initializer,
            bias_initializer="zeros"
        )
    
    def _autofhr_block(self, x, filters, kernel_size, dilation_rate, input_layer):
        """
        Defines a single block with gated activations
        
        Parameters:
        -----------
        x: tensor
            Input tensor
        filters: int
            Number of filters for the convolutions
        kernel_size: int
            Size of the convolutional kernel
        dilation_rate: int
            Dilation rate for the causal convolution
        input_layer: tensor
            Input layer for residual connection
            
        Returns:
        --------
        tuple: (output tensor, skip connection tensor)
        """
        # Gated activation unit (tanh * sigmoid)
        tanh_out = self._causal_conv1d(
            filters=filters, 
            kernel_size=kernel_size, 
            dilation_rate=dilation_rate, 
            activation='tanh'
        )(x)
        
        sigmoid_out = self._causal_conv1d(
            filters=filters,
            kernel_size=kernel_size, 
            dilation_rate=dilation_rate, 
            activation='sigmoid'
        )(x)
        
        x = Multiply()([tanh_out, sigmoid_out])

        skip_conn = self._conv1d(filters, 1)(x)

        x = Add()([x, input_layer])
        
        return x, skip_conn
    
    def build(self, input_shape, filters=40, kernel_size=20, dilation_rates=None, num_heads=8, key_dim=40):
        """
        Build the AutoFHR model with attention mechanism
        
        Parameters:
        -----------
        input_shape: tuple
            Shape of the input tensor
        filters: int
            Number of filters for the convolutions
        kernel_size: int
            Size of the convolutional kernel
        dilation_rates: list
            List of dilation rates for the blocks
            
        Returns:
        --------
        keras.Model: AutoFHR model with attention
        """
        if dilation_rates is None:
            dilation_rates = [2**i for i in range(8)]
            
        input_layer = Input(shape=input_shape)
        
        skip_connections = []
        
        x = input_layer
        
        for dilation_rate in dilation_rates:
            x, skip_conn = self._autofhr_block(x, filters, kernel_size, dilation_rate, input_layer)
            skip_connections.append(skip_conn)
            
        out = Add()(skip_connections)
        out = tf.keras.activations.sigmoid(out)
        out = keras.layers.Dropout(0.25)(out)

        out = self._conv1d(1, kernel_size=1)(out)
        out = tf.keras.activations.sigmoid(out)

        attention_out = MultiHeadAttention(num_heads, key_dim)(out, out)
        attention_out = tf.keras.activations.sigmoid(attention_out)
        out = Add()([out, attention_out])
        out = tf.keras.activations.sigmoid(out)

        out = self._conv1d(1, kernel_size=1)(out)
        out = tf.keras.activations.sigmoid(out)

        out = TimeDistributed(keras.layers.Dense(1, activation="sigmoid"))(out)

        model = Model(inputs=input_layer, outputs=out)
        
        return model 