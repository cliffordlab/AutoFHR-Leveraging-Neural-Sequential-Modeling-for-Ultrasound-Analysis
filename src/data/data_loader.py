import numpy as np
import os
from tensorflow.keras.utils import Sequence

class AutoFHRDataLoader:
    """
    Data loader
    """
    
    def __init__(self, data_path):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file (.npz)
        """
        self.data_path = data_path
        self._load_data()
        
    def _load_data(self):
        """Load data from the npz file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        print(f"Loading data from {self.data_path}...")
        try:
            loaded_data = np.load(self.data_path, allow_pickle=True)
            self.label = loaded_data['label']
            self.tensor_all = loaded_data['tensor_all']
            self.beatset = loaded_data['beatset']
            self.FHR = loaded_data['FHR']
            self.DUS = loaded_data['DUS']
            print(f"Data loaded successfully. Total samples: {len(self.tensor_all)}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def get_all_data(self):
        """
        Get all data and labels
        
        Returns:
        --------
        tuple: (data, labels)
        """
        return self.tensor_all, self.label
    
    def get_beat_sets(self):
        """
        Get beat data
        
        Returns:
        --------
        numpy.ndarray: Beat set list
        """
        return self.beatset
    
    def get_FHR_data(self):
        """
        Get FHR data
        
        Returns:
        --------
        numpy.ndarray: FHR list
        """
        return self.FHR
    
    def get_DUS_data(self):
        """
        Get DUS data
        
        Returns:
        --------
        numpy.ndarray: DUS signals
        """
        return self.DUS


class AutoFHRDataGenerator(Sequence):
    """
    Data generator for training the model
    """
    
    def __init__(self, x_data, y_data, batch_size=64, shuffle=True):
        """
        Initialize the data generator
        
        Parameters:
        -----------
        x_data : numpy.ndarray
            Input data
        y_data : numpy.ndarray
            Target data
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle the data after each epoch
        """
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x_data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Return the number of batches"""
        return int(np.ceil(len(self.x_data) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x_data[batch_indices]
        batch_y = self.y_data[batch_indices]
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices) 