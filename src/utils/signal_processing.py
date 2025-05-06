import numpy as np
from scipy.signal import butter, lfilter, resample_poly
import pywt
from skimage.transform import resize
from scipy.spatial.distance import cdist

def normalize(array):
    """
    Normalize array to range [0, 1]
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array to normalize
        
    Returns:
    --------
    numpy.ndarray: Normalized array
    """
    return (array - array.min()) / (array.max() - array.min())

def normalize_mean_std(array):
    """
    Normalize array using mean and standard deviation (z-score)
    
    Parameters:
    -----------
    array : numpy.ndarray
        Input array to normalize
        
    Returns:
    --------
    numpy.ndarray: Normalized array with zero mean and unit variance
    """
    mean = np.mean(array)
    std = np.std(array)
    normalized_array = (array - mean) / std
    return normalized_array

def select_good_quality_segments(signal, quality_array, threshold=0):
    """
    Select good quality segments from the signal based on quality array
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal
    quality_array : numpy.ndarray
        Quality array where 0 represents good quality
    threshold : float
        Threshold for quality selection (segments with quality <= threshold are kept)
        
    Returns:
    --------
    tuple: (processed_signal, quality_mask)
        - processed_signal: Signal with poor quality segments set to zeros
        - quality_mask: Boolean mask of good quality segments
    """
    if len(signal) != len(quality_array):
        raise ValueError("Signal and quality array must have the same length")
    
    # Create a mask where True represents good quality
    quality_mask = quality_array <= threshold
    
    # Create a copy of the signal
    processed_signal = signal.copy()
    
    # Set poor quality segments to zeros
    processed_signal[~quality_mask] = 0
    
    return processed_signal, quality_mask

def create_scalogram(sig, fs, time_bins, freq_bins):
    """
    Create a scalogram from a signal using continuous wavelet transform
    
    Parameters:
    -----------
    sig : numpy.ndarray
        Input signal
    fs : float
        Sampling frequency
    time_bins : int
        Number of time bins for the output scalogram
    freq_bins : int
        Number of frequency bins for the output scalogram
        
    Returns:
    --------
    numpy.ndarray: Scalogram representation of the signal
    """
    scales = np.arange(1, freq_bins + 1)
    coeffs, _ = pywt.cwt(sig, scales, wavelet='morl', sampling_period=1/fs)
    f = abs(coeffs)
    f = resize(f, (np.shape(coeffs)[0], time_bins), mode='constant')
    return f

def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Design a Butterworth bandpass filter
    
    Parameters:
    -----------
    lowcut : float
        Low cutoff frequency
    highcut : float
        High cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order
        
    Returns:
    --------
    tuple: Filter coefficients (b, a)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply a butterworth bandpass filter to data
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input signal
    lowcut : float
        Low cutoff frequency
    highcut : float
        High cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order
        
    Returns:
    --------
    numpy.ndarray: Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def DUS_filtering(DUS, fs=4000):
    """
    Apply a butterworth bandpass filter to Doppler Ultrasound Signal
    
    Parameters:
    -----------
    DUS : numpy.ndarray
        Doppler Ultrasound Signal
    fs : float
        Sampling frequency
        
    Returns:
    --------
    numpy.ndarray: Filtered DUS signal
    """
    lowcut = 25.0
    highcut = 600.0
    DUS_f = butter_bandpass_filter(DUS, lowcut, highcut, fs, order=2)
    return DUS_f

def signal_resample(signal, original_fs, target_fs):
    """
    Resample a signal from original frequency to target frequency
    
    Parameters:
    -----------
    signal : numpy.ndarray
        Input signal
    original_fs : float
        Original sampling frequency
    target_fs : float
        Target sampling frequency
        
    Returns:
    --------
    numpy.ndarray: Resampled signal
    """
    gcd = np.gcd(original_fs, target_fs)
    up = target_fs // gcd
    down = original_fs // gcd
    resampled_signal = resample_poly(signal, up, down)
    return resampled_signal

def find_closer_duplicates(beats, threshold):
    """
    Remove duplicates from beats array based on a threshold
    
    Parameters:
    -----------
    beats : numpy.ndarray
        Array of beat locations
    threshold : float
        Minimum distance between beats
        
    Returns:
    --------
    numpy.ndarray: Array of unique beats
    """
    unique_beats = []
    prev_beat = -np.inf
    for beat in sorted(beats):
        if beat - prev_beat > threshold:
            unique_beats.append(beat)
            prev_beat = beat
    return np.array(unique_beats)

def grouping_beats(beats1, beats2, fs):
    """
    Group beats from two different detectors
    
    Parameters:
    -----------
    beats1 : numpy.ndarray
        First array of beat locations
    beats2 : numpy.ndarray
        Second array of beat locations
    fs : float
        Sampling frequency
        
    Returns:
    --------
    list: Combined beats from both detectors
    """
    beats1 = find_closer_duplicates(beats1, 0.24 * fs)
    beats2 = find_closer_duplicates(beats2, 0.24 * fs)

    D = cdist(beats1[:, np.newaxis], beats2[:, np.newaxis])

    idxMinBeats1 = np.argmin(D, axis=1)
    idxMinBeats2 = np.argmin(D, axis=0)

    beatSet = []
    
    if len(np.unique(idxMinBeats1)) == len(beats1):
        for i in range(len(beats1)):
            beatSet.append(np.mean([beats1[i], beats2[idxMinBeats1[i]]]))
        
        if len(beats1) != len(beats2):
            missingBeats2 = np.setdiff1d(range(len(beats2)), idxMinBeats1)
            for i in missingBeats2:
                beatSet.append(beats2[i])
    else:
        uniqueBeats2 = np.unique(idxMinBeats1)
        
        for i in uniqueBeats2:
            idxBeat1 = np.where(idxMinBeats1 == i)[0]
            
            if len(idxBeat1) == 1:
                beatSet.append(np.mean([beats1[idxBeat1[0]], beats2[i]]))
            else:
                closerBeat1FromBeat2 = idxMinBeats2[i]
                beatSet.append(np.mean([beats1[closerBeat1FromBeat2], beats2[i]]))
                extras = beats1[np.setdiff1d(idxBeat1, closerBeat1FromBeat2)]
                beatSet.extend(extras)

    return sorted(beatSet) 