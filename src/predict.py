import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.signal_processing import DUS_filtering, create_scalogram, normalize, signal_resample

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions with a trained AutoFHR model')
    
    # Input/output parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.h5)')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input Doppler ultrasound signal file (.wav, .mat, or .npy)')
    parser.add_argument('--output_dir', type=str, default='./estimations',
                        help='Directory to save outputs')
    
    # Signal processing parameters
    parser.add_argument('--sampling_rate', type=float, default=4000,
                        help='Sampling rate of the input signal in Hz')
    parser.add_argument('--time_bins', type=int, default=1000,
                        help='Number of time bins for the scalogram')
    parser.add_argument('--freq_bins', type=int, default=40,
                        help='Number of frequency bins for the scalogram')
    
    # Other parameters
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use (use -1 for CPU)')
    
    return parser.parse_args()

def setup_gpu(gpu_id):
    """Set up GPU device"""
    if gpu_id >= 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                # Allow memory growth for the GPU
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                print(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
        else:
            print("No GPUs found. Using CPU instead.")
    else:
        # Hide all GPUs
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU as specified.")

def load_signal(input_file, sampling_rate=4000):
    """
    Load a Doppler ultrasound signal from a file
    
    Parameters:
    -----------
    input_file : str
        Path to the input file (.wav, .mat, or .npy)
    sampling_rate : float
        Expected sampling rate of the signal
        
    Returns:
    --------
    numpy.ndarray: Loaded signal
    """
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.wav':
        # Load from WAV file
        import scipy.io.wavfile as wavfile
        sr, signal = wavfile.read(input_file)
        if sr != sampling_rate:
            print(f"Warning: Sampling rate of WAV file ({sr} Hz) does not match expected rate ({sampling_rate} Hz)")
        
        # If stereo, take only the first channel
        if len(signal.shape) > 1:
            signal = signal[:, 0]
            
    elif file_ext == '.mat':
        # Load from MATLAB file
        mat_data = loadmat(input_file)
        # Try to find the signal in the mat file
        signal = None
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray) and value.size > 100 and not key.startswith('__'):
                signal = value.squeeze()
                break
        
        if signal is None:
            raise ValueError(f"Could not find a suitable signal array in the MATLAB file")
            
    elif file_ext == '.npy':
        # Load from NumPy file
        signal = np.load(input_file)
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return signal

def preprocess_signal(signal, sampling_rate, time_bins, freq_bins):
    """
    Preprocess the signal for input to the model
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal
    sampling_rate : float
        Sampling rate of the signal
    time_bins : int
        Number of time bins for the scalogram
    freq_bins : int
        Number of frequency bins for the scalogram
        
    Returns:
    --------
    numpy.ndarray: Preprocessed signal ready for model input
    """
    # Filter the signal
    filtered_signal = normalize(DUS_filtering(signal_resample(signal, 4000, fs=sampling_rate), fs=sampling_rate))
    scalogram = create_scalogram(filtered_signal, sampling_rate, time_bins, freq_bins)
    normalized_scalogram = normalize(scalogram)
    
    model_input = normalized_scalogram.reshape(1, *normalized_scalogram.shape)
    
    return model_input, filtered_signal

def main():
    """Main function to make predictions with a trained AutoFHR model"""
    args = parse_args()
    
    # Set up GPU
    setup_gpu(args.gpu)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    
    # Load the signal
    print(f"Loading signal from {args.input_file}")
    signal = load_signal(args.input_file, args.sampling_rate)
    print(f"Signal loaded with shape: {signal.shape}")
    
    # Preprocess the signal
    print("Preprocessing signal...")
    model_input, filtered_signal = preprocess_signal(
        signal, args.sampling_rate, args.time_bins, args.freq_bins
    )
    
    # Make prediction
    print("Making prediction...")
    prediction = model.predict(model_input)[0]

    # Save the results
    results = {
        'original_signal': signal,
        'filtered_signal': filtered_signal,
        'prediction': prediction,
    }
    
    np.save(os.path.join(args.output_dir, 'prediction_results.npy'), results)
    savemat(os.path.join(args.output_dir, 'prediction_results.mat'), results)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main() 