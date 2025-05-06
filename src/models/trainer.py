import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import time
from datetime import datetime

class ModelTrainer:
    """
    Model trainer
    """
    
    def __init__(self, model, optimizer='adam', loss=None, metrics=None, checkpoint_dir='./checkpoints'):
        """
        Initialize the model trainer
        
        Parameters:
        -----------
        model : keras.Model
            Model to train
        optimizer : str or keras.optimizers.Optimizer
            Optimizer for training
        loss : function or str
            Loss function for training
        metrics : list
            List of metrics to track
        checkpoint_dir : str
            Directory to save model checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss if loss is not None else 'binary_crossentropy'
        self.metrics = metrics if metrics is not None else ['val_loss']
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
    def setup_callbacks(self, patience=100, log_dir=None, model_name="model"):
        """
        Set up callbacks for training
        
        Parameters:
        -----------
        patience : int
            Number of epochs with no improvement after which training will be stopped
        log_dir : str
            Directory for the logs
        model_name : str
            Name of the model for saving checkpoints
            
        Returns:
        --------
        list: List of callbacks
        """
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor=self.metrics,
            save_best_only=True,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor=self.metrics,
            patience=patience,
            mode='min',
            verbose=1,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # TensorBoard callback
        if log_dir is not None:
            # Create a timestamped log directory
            if log_dir is True:
                log_dir = os.path.join('logs', f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            
            os.makedirs(log_dir, exist_ok=True)
            tensorboard = TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
            
        return callbacks
    
    def train(self, train_data, train_labels, val_data, val_labels, epochs=1000, batch_size=64, 
              callbacks=None, verbose=1):
        """
        Train the model
        
        Parameters:
        -----------
        train_data : numpy.ndarray
            Training data
        train_labels : numpy.ndarray
            Training labels
        val_data : numpy.ndarray
            Validation data
        val_labels : numpy.ndarray
            Validation labels
        epochs : int
            Number of epochs to train
        batch_size : int
            Batch size for training
        callbacks : list
            List of callbacks for training
        verbose : int
            Verbosity mode (0, 1, or 2)
            
        Returns:
        --------
        keras.callbacks.History: Training history
        """
        if callbacks is None:
            callbacks = self.setup_callbacks()
            
        start_time = time.time()
        
        # Train the model
        history = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        return history
    
    def train_with_generator(self, train_generator, val_generator, epochs=1000, callbacks=None, verbose=1):
        """
        Train the model using data generators
        
        Parameters:
        -----------
        train_generator : keras.utils.Sequence
            Training data generator
        val_generator : keras.utils.Sequence
            Validation data generator
        epochs : int
            Number of epochs to train
        callbacks : list
            List of callbacks for training
        verbose : int
            Verbosity mode (0, 1, or 2)
            
        Returns:
        --------
        keras.callbacks.History: Training history
        """
        if callbacks is None:
            callbacks = self.setup_callbacks()
            
        start_time = time.time()
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        return history
    
    def evaluate(self, test_data, test_labels, batch_size=64, verbose=1):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        test_data : numpy.ndarray
            Test data
        test_labels : numpy.ndarray
            Test labels
        batch_size : int
            Batch size for evaluation
        verbose : int
            Verbosity mode (0, 1, or 2)
            
        Returns:
        --------
        list: Test metrics
        """
        return self.model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=verbose)
    
    def predict(self, data, batch_size=64, verbose=0):
        """
        Make predictions with the model
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data
        batch_size : int
            Batch size for prediction
        verbose : int
            Verbosity mode (0, 1, or 2)
            
        Returns:
        --------
        numpy.ndarray: Model predictions
        """
        return self.model.predict(data, batch_size=batch_size, verbose=verbose)
    
    def save_model(self, filepath):
        """
        Save the model to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_weights(self, filepath):
        """
        Load model weights from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the weights file
        """
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}") 