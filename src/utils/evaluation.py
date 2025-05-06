import numpy as np
import matplotlib.pyplot as plt

def prediction_with_ground_truth(signal, ground_truth, prediction, detected_peaks=None, 
                                      ground_truth_peaks=None, figsize=(15, 8)):
    """
    Plot signal with ground truth and prediction
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal
    ground_truth : numpy.ndarray
        Ground truth labels
    prediction : numpy.ndarray
        Model predictions
    detected_peaks : numpy.ndarray, optional
        Detected peak locations
    ground_truth_peaks : numpy.ndarray, optional
        Ground truth peak locations
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot the input signal
    ax[0].plot(signal, label='Input Signal')
    ax[0].set_title('Input Signal')
    ax[0].legend()
    
    # Plot ground truth
    ax[1].plot(ground_truth, label='Ground Truth')
    if ground_truth_peaks is not None:
        ax[1].plot(ground_truth_peaks, ground_truth[ground_truth_peaks], 'ro', label='GT Peaks')
    ax[1].set_title('Ground Truth')
    ax[1].legend()
    
    # Plot prediction
    ax[2].plot(prediction, label='Estimation')
    if detected_peaks is not None:
        ax[2].plot(detected_peaks, prediction[detected_peaks], 'go', label='Detected Peaks')
    ax[2].set_title('Model Estimation')
    ax[2].legend()
    
    plt.tight_layout()
    return fig