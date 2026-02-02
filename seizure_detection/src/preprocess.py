import numpy as np
import scipy.signal
import torch

TARGET_FREQ = 250  #Hz

def resample_signal(data, original_sf, target_sf=TARGET_FREQ):
    """
    Resample the signal from original_sf to target_sf.
    Args:
        data: (Channels, Time) numpy array
        original_sf: int/float
        target_sf: int/float
    Returns:
        resampled_data: (Channels, New_Time)
    """
    if original_sf == target_sf:
        return data
    
    n_samples = data.shape[1]
    duration = n_samples / original_sf
    new_n_samples = int(duration * target_sf)
    
    # Scipy resample uses FFT, good for periodic. 
    # resample_poly is better for non-periodic but slower. 
    # For speed/simplicity here using resample.
    resampled_data = scipy.signal.resample(data, new_n_samples, axis=1)
    return resampled_data

def normalize_signal(data):
    """
    Z-score normalization per channel.
    Args:
        data: (Channels, Time)
    Returns:
        normalized_data: (Channels, Time)
    """
    # Mean and Std per channel
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    
    # Avoid div by zero
    std[std == 0] = 1.0
    
    return (data - mean) / std

def apply_filter(data, sf, low=0.5, high=40.0, notch=60.0):
    """
    Apply Band-pass and Notch filter.
    data: (Channels, Time)
    """
    # 1. Band-pass
    nyquist = 0.5 * sf
    low_norm = low / nyquist
    high_norm = high / nyquist
    # Ensure high is within valid range
    if high_norm >= 1.0: high_norm = 0.99
    
    b, a = scipy.signal.butter(4, [low_norm, high_norm], btype='band')
    data = scipy.signal.filtfilt(b, a, data, axis=1)
    
    # 2. Notch (Line noise) - Optional but good practice
    # Usually 60Hz (US) or 50Hz (EU). If sf < notch*2, we can't notch.
    if sf > notch * 2:
        bn, an = scipy.signal.iirnotch(notch, 30.0, sf)
        data = scipy.signal.filtfilt(bn, an, data, axis=1)
        
    return data

def preprocess_pipeline(raw_data, sf):
    """
    Full pipeline: Filter -> Resample -> Normalize
    """
    # Filter FIRST (on high res data)
    data = apply_filter(raw_data, sf, low=0.5, high=40.0)
    
    # Resample
    data = resample_signal(data, sf, TARGET_FREQ)
    
    # Normalize
    data = normalize_signal(data)
    return data
