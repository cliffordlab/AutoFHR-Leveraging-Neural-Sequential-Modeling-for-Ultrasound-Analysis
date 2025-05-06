import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from matplotlib.patches import Rectangle
import pywt
from .signal_processing import normalize

def bland_altman_plot(reference, test, title="Bland-Altman Plot", figsize=(10, 8), save_path=None):
    """
    Create a Bland-Altman plot to compare two measurement methods
    
    Parameters:
    -----------
    reference : numpy.ndarray
        Reference measurements (ground truth)
    test : numpy.ndarray
        Test measurements (predictions)
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    tuple: (fig, ax) Matplotlib figure and axis objects
    """
    # Convert to numpy arrays if they aren't already
    reference = np.asarray(reference)
    test = np.asarray(test)
    
    # Calculate mean and difference between methods
    mean = np.mean([reference, test], axis=0)
    diff = test - reference
    
    # Calculate mean difference and standard deviation of differences
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter plot of differences vs means
    ax.scatter(mean, diff, alpha=0.7)
    
    # Plot the mean difference line
    ax.axhline(md, color='red', linestyle='-', label=f'Mean diff: {md:.2f}')
    
    # Plot limits of agreement (LoA) - 1.96*SD
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--', 
               label=f'Upper LoA: {(md + 1.96*sd):.2f}')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--', 
               label=f'Lower LoA: {(md - 1.96*sd):.2f}')
    
    # Add labels and title
    ax.set_xlabel('Mean of Reference and Test')
    ax.set_ylabel('Difference (Test - Reference)')
    ax.set_title(title)
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def scatter_plot_with_regression(reference, test, title="Scatter Plot with Regression Line", 
                                figsize=(10, 8), save_path=None):
    """
    Create a scatter plot with regression line to compare reference and test values
    
    Parameters:
    -----------
    reference : numpy.ndarray
        Reference measurements (ground truth)
    test : numpy.ndarray
        Test measurements (predictions)
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    tuple: (fig, ax) Matplotlib figure and axis objects
    """
    # Convert to numpy arrays if they aren't already
    reference = np.asarray(reference)
    test = np.asarray(test)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter plot
    ax.scatter(reference, test, alpha=0.7)
    
    # Calculate correlation coefficient
    correlation, p_value = stats.pearsonr(reference, test)
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(reference, test)
    x = np.linspace(min(reference), max(reference), 100)
    y = slope * x + intercept
    
    # Plot regression line
    ax.plot(x, y, 'r-', label=f'y = {slope:.3f}x + {intercept:.3f}')
    
    # Add identity line (y=x)
    ax.plot([min(reference), max(reference)], [min(reference), max(reference)], 
            'k--', alpha=0.5, label='Identity Line')
    
    # Add labels and title
    ax.set_xlabel('Reference')
    ax.set_ylabel('Test')
    ax.set_title(f"{title}\nCorrelation: {correlation:.3f}, R²: {r_value**2:.3f}")
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def attention_plot(signal, attention_weights, time_axis=None, figsize=(12, 8), 
                  save_path=None, title="Attention Visualization"):
    """
    Visualize attention weights over a signal
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal
    attention_weights : numpy.ndarray
        Attention weights
    time_axis : numpy.ndarray, optional
        Time axis for the signal
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    title : str
        Title for the plot
        
    Returns:
    --------
    tuple: (fig, axes) Matplotlib figure and axes objects
    """
    if time_axis is None:
        time_axis = np.arange(len(signal))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot signal
    axes[0].plot(time_axis, signal, 'b-', label='Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot attention weights as heatmap
    im = axes[1].imshow(attention_weights.reshape(1, -1), aspect='auto', 
                       extent=[time_axis[0], time_axis[-1], 0, 1], 
                       cmap='viridis', alpha=0.8)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[1], orientation='vertical', pad=0.01)
    cbar.set_label('Attention Weight')
    
    # Plot attention weights as line
    axes[1].plot(time_axis, attention_weights, 'r-', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Attention')
    axes[1].set_ylim([0, attention_weights.max() * 1.1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def time_series_robustness_plot(original_signal, perturbed_signals, original_prediction, 
                               perturbed_predictions, time_axis=None, figsize=(15, 10), 
                               save_path=None, title="Time Series Robustness Analysis"):
    """
    Plot original and perturbed time series with their predictions to visualize robustness
    
    Parameters:
    -----------
    original_signal : numpy.ndarray
        Original input signal
    perturbed_signals : list of numpy.ndarray
        List of perturbed versions of the input signal
    original_prediction : numpy.ndarray
        Prediction on the original signal
    perturbed_predictions : list of numpy.ndarray
        Predictions on the perturbed signals
    time_axis : numpy.ndarray, optional
        Time axis for the signals
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    title : str
        Title for the plot
        
    Returns:
    --------
    tuple: (fig, axes) Matplotlib figure and axes objects
    """
    n_perturbations = len(perturbed_signals)
    
    if time_axis is None:
        time_axis = np.arange(len(original_signal))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_perturbations + 1, 2, figsize=figsize, sharex='col')
    
    # Plot original signal and prediction
    axes[0, 0].plot(time_axis, original_signal, 'b-')
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_axis, original_prediction, 'g-')
    axes[0, 1].set_title('Original Prediction')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot perturbed signals and their predictions
    for i in range(n_perturbations):
        axes[i+1, 0].plot(time_axis, perturbed_signals[i], 'r-')
        axes[i+1, 0].set_ylabel(f'Perturbed {i+1}')
        axes[i+1, 0].grid(True, alpha=0.3)
        
        # Calculate and show difference from original signal
        diff = np.mean(np.abs(perturbed_signals[i] - original_signal))
        axes[i+1, 0].set_title(f'Perturbed Signal {i+1} (Diff: {diff:.4f})')
        
        axes[i+1, 1].plot(time_axis, perturbed_predictions[i], 'r-')
        
        # Calculate and show difference from original prediction
        pred_diff = np.mean(np.abs(perturbed_predictions[i] - original_prediction))
        axes[i+1, 1].set_title(f'Perturbed Prediction {i+1} (Diff: {pred_diff:.4f})')
        axes[i+1, 1].grid(True, alpha=0.3)
    
    # Set common labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Time')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def plot_predictions_with_ground_truth(signals, predictions, ground_truths, labels=None, 
                                     figsize=(15, 10), n_samples=4, save_path=None, 
                                     title="Predictions vs Ground Truth"):
    """
    Plot predictions alongside ground truth for multiple signals
    
    Parameters:
    -----------
    signals : numpy.ndarray
        Input signals, shape (n_samples, signal_length)
    predictions : numpy.ndarray
        Predicted values, shape (n_samples, signal_length)
    ground_truths : numpy.ndarray
        Ground truth values, shape (n_samples, signal_length)
    labels : list of str, optional
        Labels for each sample
    figsize : tuple
        Figure size (width, height)
    n_samples : int, optional
        Number of samples to plot
    save_path : str, optional
        Path to save the figure
    title : str
        Title for the plot
        
    Returns:
    --------
    tuple: (fig, axes) Matplotlib figure and axes objects
    """
    n_samples = min(n_samples, len(signals))
    
    if labels is None:
        labels = [f"Sample {i+1}" for i in range(n_samples)]
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_samples, 1, figsize=figsize, sharex=True)
    
    # If only one sample is plotted, axes will not be an array
    if n_samples == 1:
        axes = [axes]
    
    # Loop through the samples
    for i in range(n_samples):
        ax = axes[i]
        
        # Time axis
        time_axis = np.arange(len(signals[i]))
        
        # Plot signal
        ax.plot(time_axis, signals[i], 'b-', alpha=0.5, label='Signal')
        
        # Plot ground truth (using step to visualize peaks)
        ax.step(time_axis, ground_truths[i], 'g-', where='mid', 
                label='Ground Truth', alpha=0.7)
        
        # Plot prediction
        ax.plot(time_axis, predictions[i], 'r-', label='Prediction', alpha=0.7)
        
        # Add sample label
        ax.set_title(labels[i])
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Add legend (only to the first plot to save space)
        if i == 0:
            ax.legend(loc='upper right')
    
    # Add common x-label
    axes[-1].set_xlabel('Time')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes 

def plot_scalogram(scalogram, times=None, frequencies=None, dus_signal=None, 
                   gold_standard=None, estimated=None, attention_weights=None, 
                   fs=4000, figsize=(12, 5), save_path=None, 
                   title="Scalogram Visualization", cmap='plasma'):
    """
    Plot a scalogram with optional overlays: DUS signal, gold standard, estimated labels, and attention weights.

    Parameters:
    -----------
    scalogram : numpy.ndarray
        Scalogram data (frequency x time)
    times : numpy.ndarray, optional
        Time axis values (in seconds)
    frequencies : numpy.ndarray, optional
        Frequency axis values (in Hz)
    dus_signal : numpy.ndarray, optional
        Doppler Ultrasound Signal to overlay
    gold_standard : numpy.ndarray, optional
        Gold standard labels to overlay
    estimated : numpy.ndarray, optional
        Estimated labels to overlay
    attention_weights : numpy.ndarray, optional
        Attention weights to overlay
    fs : int, optional
        Sampling frequency for frequency calculation
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap for the scalogram

    Returns:
    --------
    tuple: (fig, axes) Matplotlib figure and list of axes objects
    """

    # Initialize time axis if not provided (in seconds)
    if times is None:
        times = np.linspace(0, 3.75, scalogram.shape[1])

    # Initialize frequency axis if not provided
    if frequencies is None:
        scales = np.arange(1, scalogram.shape[0] + 1)
        fc = pywt.central_frequency('morl')
        frequencies = fc * fs / scales

    # Create figure
    fig, ax_main = plt.subplots(figsize=figsize)

    # Plot the scalogram
    extent = [times[0], times[-1], 0, len(frequencies)-1]
    ax_main.imshow(scalogram, extent=extent, cmap=cmap, aspect='auto', origin='lower')
    yticks = np.arange(0, len(frequencies), 4)
    ytick_labels = [f"{frequencies[i]:.0f}" for i in yticks]
    ax_main.set_yticks(np.arange(len(frequencies)))
    ax_main.set_yticklabels([f"{freq:.0f}" for freq in frequencies])
    ax_main.set_xlabel('Time (min)', fontsize=15)
    ax_main.set_ylabel('Frequency (Hz)', fontsize=15)
    ax_main.set_yticks(yticks)
    ax_main.set_yticklabels(ytick_labels)
    ax_main.set_title(title)

    # Invert y-axis to display high frequencies at bottom
    ax_main.invert_yaxis()
    ax_main.grid(False)

    axes = [ax_main]

    # Overlay Attention Weights if provided
    if attention_weights is not None:
        ax_attention = ax_main.twinx()
        ax_attention.imshow(
            attention_weights[np.newaxis, :],
            aspect='auto', extent=[times[0], times[-1], 0, 1],
            cmap='gray', alpha=0.3
        )
        ax_attention.set_ylim(0, 1)
        ax_attention.axis('off')
        axes.append(ax_attention)

    # Overlay Gold Standard labels if provided
    if gold_standard is not None:
        ax_gold = ax_main.twinx()
        ax_gold.plot(times, gold_standard/2, color='pink', linewidth=2, label='Gold Standard')
        ax_gold.set_ylim(np.min(gold_standard)-0.1, np.max(gold_standard)+0.1)
        ax_gold.axis('off')
        axes.append(ax_gold)

    # Overlay Estimated labels if provided
    if estimated is not None:
        ax_estimated = ax_main.twinx()
        ax_estimated.plot(times, estimated/2, color='orange', linewidth=2, label='Estimated')
        ax_estimated.set_ylim(np.min(estimated)-0.1, np.max(estimated)+0.1)
        ax_estimated.axis('off')
        axes.append(ax_estimated)

    # Overlay DUS signal if provided
    if dus_signal is not None:
        dus_times = np.linspace(times[0], times[-1], len(dus_signal))
        ax_dus = ax_main.twinx()
        ax_dus.plot(dus_times, dus_signal/2, color='darkkhaki', linewidth=1.5, label='DUS Signal')
        ax_dus.set_ylim(np.min(dus_signal), np.max(dus_signal))
        ax_dus.axis('off')
        axes.append(ax_dus)

    # Add legend (handles all possible overlays)
    lines_labels = [ax.get_legend_handles_labels() for ax in axes[1:]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    if lines:
        fig.legend(lines, labels, loc='upper left', framealpha=1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, axes

def play_audio(file_path, rate=None):
    """
    Play audio file in Jupyter notebook
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    rate : int, optional
        Playback rate (if None, use the file's original rate)
        
    Returns:
    --------
    IPython.display.Audio: Audio player widget
    """
    from IPython.display import Audio
    return Audio(file_path, rate=rate) 