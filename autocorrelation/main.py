import os
import librosa
import Ipython.display as ipd
import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy import signal
from scipy.io import wavfile
import sys
from statsmodels.tsa.stattools import acf

'''
This script is designed to estimate heart rates from Doppler signals using signal processing techniques. 
It includes functions for filtering, envelope extraction, autocorrelation, and peak detection. 
The script takes Doppler files as input, processes the Doppler data, and estimates the heart rate present in the Doppler.
The details of the model performance and parameter optimization methods are available in the following paper:

Valderrama CE, Stroux L, Katebi N, Paljug E, Hall-Clifford R, Rohloff P, Marzbanrad F, Clifford GD. 
An open source autocorrelation-based method for fetal heart rate estimation from one-dimensional Doppler ultrasound. 
Physiological measurement. 2019 Feb 25;40(2):025005.
Link to the Matlab code: https://github.com/cliffordlab/fhr_dus1/blob/master/getFHR.m

Functions:
- butter_lowpass_filter: Implements a low-pass Butterworth filter on input data.
- butter_highpass_filter: Implements a high-pass Butterworth filter on input data.
- hilbert_filter: Computes the amplitude envelope of a signal using the Hilbert transform.
- autocorrelate: Computes the autocorrelation of a signal.
- estimate_heart_rate: Estimates the heart rate from a filtered and processed Doppler signal.
- main: The main function that loads an Doppler file, processes it, and estimates the heart rate.

Usage:
- Run the script from the command line with the name of an Doppler file as a command-line argument.
- The script will process the Doppler file, estimate the heart rate, and print the result.

Authors: Aishwarya Shashikumar and Nasim Katebi, July 2023

Note:
- The script requires 3.75 s Doppler file in WAV format to be present in the same directory or provide the correct path to it.
- The autocorrelation function calculation was switched from using Librosa to Statsmodels version 0.14.1 to ensure consistency with the MATLAB autocorrelation function results. (Mohsen Motieshirazi, Feb 2024)
'''



# Define a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order):
#'data': input signal to be filtered, 
# 'cutoff': cut-off frequency of the low-pass filter, 
# 'fs': sampling frequency of the input signal,
# 'order': the order of the filter. 
# 'sos':filter coefficients.   
	t = np.arange(len(data))/fs
	sos = signal.butter(order, cutoff, btype='low',fs=fs,output='sos', analog=False)
	y = signal.sosfiltfilt(sos, data) #applies the filter to the input signal
	y = np.asarray(y)

	return y
# Define a high-pass Butterworth filter
def butter_highpass_filter(data, cutoff, fs, order):
	sos = signal.butter(order, cutoff, btype='high',fs=fs,output='sos', analog=False)
	y = signal.sosfiltfilt(sos, data)
	return y

# Define a function to compute the amplitude envelope using the Hilbert transform
def hilbert_filter(x):
	analytic_signal = hilbert(x)
	cutoff=14.794675954494590
	amplitude_envelope = np.exp(butter_lowpass_filter(np.log(np.abs(analytic_signal)),cutoff,4000,2))

	return amplitude_envelope


# Define a function to compute autocorrelation
def autocorrelate(data):
	
	#autocor = librosa.autocorrelate(data)
    autocor = acf(data, nlags=len(data)-1)

    return autocor



def estimate_heart_rate(x, Fs):
	lpf = butter_lowpass_filter(x,600,Fs,4)
	hpf = butter_highpass_filter(lpf,25,Fs,4)
	r = hilbert_filter(hpf)
	corsig = autocorrelate(r)

	minPeriod = 0.287308661149887 #Lower interval to start searching peaks in the AC function (s)
	maxPeriod = 0.839140444683137 #Upper interval to start searching peaks in the AC function (s)
	min_index = round(minPeriod * Fs)
	max_index = round(maxPeriod * Fs)
	ratioFirstPeak = 0.650124394601487
	seg=corsig[min_index:max_index]
	locs, _ = find_peaks(seg, distance=(1/3)*Fs)

	peaks=seg[locs]

	if locs is not None:
		if len(peaks)>1:
			perX = (min_index+locs[0]-1)/(min_index+locs[1]-1)

			if perX>=0.48 and perX<=.52:
				index = locs[0]
			else:
				perDif = peaks[0]/peaks[1]

				if perDif<0:
					idxMaxPk = np.argmax(peaks)
					index = locs[idxMaxPk]
				else:
					if perDif >= ratioFirstPeak:
						index = locs[0]
					else:
						index = locs[1]
		else:
			index = locs[0]

		true_index = index+min_index-1

		heartRate = 60/(true_index/Fs)
	else:
		heartRate = float('nan')
	return heartRate

def main(Doppler_file,Fs):
    try:
        # Load the Doppler data based on the provided input
        Fs, x = wavfile.read(Doppler_file)

        # Estimate the heart rate using your functions
        heart_rate = estimate_heart_rate(x, Fs)

        if not np.isnan(heart_rate):
            print("Estimated Heart Rate: {:.2f} bpm".format(heart_rate))
            return heart_rate
        else:
            print("Heart rate estimation failed.")
    except Exception as e:
        print("An error occurred:", e)