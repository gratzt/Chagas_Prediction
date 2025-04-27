# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 11:37:13 2025

@author: trevo
"""

import numpy as np
import os
import pywt
from scipy import signal
import wfdb
import torch
from scipy import interpolate

class ECGPreprocess:
    def __init__(self, sampling_rate=400, highpass_freq=0.5, lowpass_freq=100, 
                 wavelet='sym6', wavelet_level=8):
        """
        Initialize ECG filtering class with parameters optimized for ECG processing.
        
        Parameters:
        -----------
        sampling_rate : int
            Target sampling rate in Hz
        highpass_freq : float
            High-pass filter cutoff frequency in Hz
        lowpass_freq : float
            Low-pass filter cutoff frequency in Hz
        wavelet : str
            Wavelet family to use for decomposition
        wavelet_level : int
            Level of wavelet decomposition
        """
        self.sampling_rate = sampling_rate
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
    
    def resample_signal(self, signal_data, original_fs):
        """
        Resample the ECG signal to the target sampling rate.
        
        Parameters:
        -----------
        signal_data : numpy.ndarray
            ECG signal data
        original_fs : int
            Original sampling frequency of the signal
            
        Returns:
        --------
        numpy.ndarray
            Resampled signal
        """
        if original_fs == self.sampling_rate:
            return signal_data
        
        # Calculate number of samples for target frequency
        num_samples = int(len(signal_data) * self.sampling_rate / original_fs)
        
        # Create time arrays for original and target signals
        t_original = np.arange(len(signal_data)) / original_fs
        t_resampled = np.arange(num_samples) / self.sampling_rate
        
        # Resample using interpolation
        if len(signal_data.shape) == 1:  # Single lead
            f = interpolate.interp1d(t_original, signal_data, kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
            resampled = f(t_resampled)
        else:  # Multiple leads
            resampled = np.zeros((num_samples, signal_data.shape[1]))
            for i in range(signal_data.shape[1]):
                f = interpolate.interp1d(t_original, signal_data[:, i], kind='cubic', 
                                        bounds_error=False, fill_value='extrapolate')
                resampled[:, i] = f(t_resampled)
                
        return resampled
    
    def apply_highpass_filter(self, signal_data):
        """
        Apply high-pass filter to remove baseline wander.
        
        Parameters:
        -----------
        signal_data : numpy.ndarray
            ECG signal data
            
        Returns:
        --------
        numpy.ndarray
            Filtered signal
        """
        nyquist = self.sampling_rate / 2
        cutoff = self.highpass_freq / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(3, cutoff, btype='highpass')
        
        # Apply zero-phase filtering
        if len(signal_data.shape) == 1:  # Single lead
            filtered = signal.filtfilt(b, a, signal_data)
        else:  # Multiple leads
            filtered = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                filtered[:, i] = signal.filtfilt(b, a, signal_data[:, i])
                
        return filtered
    
    def apply_lowpass_filter(self, signal_data):
        """
        Apply low-pass filter to remove high-frequency noise.
        
        Parameters:
        -----------
        signal_data : numpy.ndarray
            ECG signal data
            
        Returns:
        --------
        numpy.ndarray
            Filtered signal
        """
        nyquist = self.sampling_rate / 2
        cutoff = self.lowpass_freq / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(4, cutoff, btype='lowpass')
        
        # Apply zero-phase filtering
        if len(signal_data.shape) == 1:  # Single lead
            filtered = signal.filtfilt(b, a, signal_data)
        else:  # Multiple leads
            filtered = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                filtered[:, i] = signal.filtfilt(b, a, signal_data[:, i])
                
        return filtered
    
    def apply_wavelet_filter(self, signal_data):
        """
        Apply wavelet-based denoising to the ECG signal.
        
        Parameters:
        -----------
        signal_data : numpy.ndarray
            ECG signal data
            
        Returns:
        --------
        numpy.ndarray
            Wavelet filtered signal
        """
        if len(signal_data.shape) == 1:  # Single lead
            # Decompose signal
            coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.wavelet_level)
            
            # Modify coefficients
            # Attenuate but not completely remove approximation coefficients
            coeffs[0] = coeffs[0] * 0.2
            
            # Apply thresholding to detail coefficients
            for i in range(1, len(coeffs)):
                sigma = np.median(np.abs(coeffs[i])) / 0.6745
                
                # Adjust threshold level based on importance for ECG features
                if i <= 3:  # Levels containing QRS and important Chagas features
                    threshold = sigma * 1.5  # Conservative threshold
                else:
                    threshold = sigma * 2.5  # More aggressive noise removal
                    
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            
            # Reconstruct signal
            filtered = pywt.waverec(coeffs, self.wavelet)
            filtered = filtered[:len(signal_data)]  # Match original length
        
        else:  # Multiple leads
            filtered = np.zeros_like(signal_data)
            for i in range(signal_data.shape[1]):
                # Process each lead independently
                lead_filtered = self.apply_wavelet_filter(signal_data[:, i])
                filtered[:, i] = lead_filtered
                
        return filtered
    
    def process_wfdb_files(self, record, apply_resample=False, apply_highpass=False, 
                          apply_lowpass=False, apply_wavelet=False, pad_to_length=None,
                          device='cpu'):
        """
        Process a list of WFDB ECG recordings and apply selected filters.
        
        Parameters:
        -----------
        record : wfdb file read in
            WFDB record file read in
        apply_resample : bool
            Whether to apply resampling
        apply_highpass : bool
            Whether to apply high-pass filtering
        apply_lowpass : bool
            Whether to apply low-pass filtering
        apply_wavelet : bool
            Whether to apply wavelet filtering
        pad_to_length : int or None
            Length to pad/truncate all signals to. If None, no padding is applied.
            
        Returns:
        --------
        torch.Tensor
            Tensor of processed ECG signals with shape (N, seq_length, 12)
        """

        
        # Get signal data and arrange to have leads in columns
        signal_data = record.p_signal
        original_fs = record.fs
        
        # Ensure we have 12 leads
        if signal_data.shape[1] != 12:
            print(f"Warning: Record does not have 12 leads. Skipping.")
            
        
        # Apply processing chain
        if apply_resample and original_fs != self.sampling_rate:
            signal_data = self.resample_signal(signal_data, original_fs)
        
        if apply_highpass:
            signal_data = self.apply_highpass_filter(signal_data)
        
        if apply_lowpass:
            signal_data = self.apply_lowpass_filter(signal_data)
        
        if apply_wavelet:
            signal_data = self.apply_wavelet_filter(signal_data)
        
                
        # Handle padding/truncation if requested
        if pad_to_length is not None:
            if signal_data.shape[0] > pad_to_length:
                # Truncate
                signal_data = signal_data[:pad_to_length, :]
            elif signal_data.shape[0] < pad_to_length:
                # Pad with zeros
                padding = np.zeros((pad_to_length - signal_data.shape[0], signal_data.shape[1]))
                signal_data = np.vstack((signal_data, padding))
            else:
                # Already correct length
                pass
            
            # Convert to tensor - shape will be N x seq_length x 12
            result_tensor = torch.tensor(np.array(signal_data), dtype=torch.float32)
        
        result_tensor = result_tensor.to(device)
        return result_tensor

