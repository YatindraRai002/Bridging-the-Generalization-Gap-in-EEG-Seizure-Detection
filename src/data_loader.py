import numpy as np
import scipy.io
import scipy.signal
import torch

def load_eeg_file(file_path):
    """
    Loads EEG data from a .mat file.
    Returns:
        raw_eeg (np.array): Shape (channels, time)
        fs (float): Sampling rate
    """

    try:
        mat = scipy.io.loadmat(file_path)

        if 'data' in mat and 'freq' in mat:
            raw_eeg = mat['data']

            if mat['freq'].size == 1:
                fs = mat['freq'].item()
            else:
                fs = mat['freq'][0][0]
        elif 'dataStruct' in mat:
            ds = mat['dataStruct'][0, 0]
            raw_eeg = ds['data'].T
            fs = ds['iEEGsamplingRate'][0][0]
        else:

            valid_keys = [k for k in mat.keys() if not k.startswith('__')]
            if len(valid_keys) > 0:

                 raise ValueError(f"Unknown structure. Keys: {valid_keys}")
            raise ValueError(f"Unknown .mat structure in {file_path}")

        return raw_eeg.astype(np.float32), float(fs)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def apply_notch_filter(eeg, fs, freq=60.0, quality=30):
    """Applies a notch filter to remove power line noise (60Hz)."""
    b, a = scipy.signal.iirnotch(freq, quality, fs)
    filtered_eeg = scipy.signal.filtfilt(b, a, eeg, axis=1)
    return filtered_eeg

def apply_bandpass_filter(eeg, fs, low=1.0, high=70.0, order=4):
    """Applies a bandpass filter to keep relevant EEG frequencies."""
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_eeg = scipy.signal.filtfilt(b, a, eeg, axis=1)
    return filtered_eeg

def normalize_channel_wise(eeg):
    """
    Z-score normalization per channel.
    eeg: (channels, time)
    """
    mean = np.mean(eeg, axis=1, keepdims=True)
    std = np.std(eeg, axis=1, keepdims=True)
    return (eeg - mean) / (std + 1e-6)

def generate_spectrogram(eeg, fs, nperseg=256, noverlap=224):
    """
    Generates a spectrogram for each channel.
    eeg: (channels, time)
    Returns:
        specs: (channels, freq, time)
    """
    specs = []
    for ch_idx in range(eeg.shape[0]):
        f, t, Sxx = scipy.signal.spectrogram(eeg[ch_idx], fs=fs, nperseg=nperseg, noverlap=noverlap)

        Sxx = np.log1p(Sxx)
        specs.append(Sxx)

    return np.array(specs)

def preprocess_sample(file_path):
    """
    Full pipeline: Load -> Filter -> Normalize -> Spectrogram
    Returns:
        raw_tensor (torch.Tensor): (channels, time)
        spec_tensor (torch.Tensor): (channels, freq, time)
    """
    raw_eeg, fs = load_eeg_file(file_path)
    if raw_eeg is None:
        return None, None

    eeg = apply_notch_filter(raw_eeg, fs)
    eeg = apply_bandpass_filter(eeg, fs)

    eeg_norm = normalize_channel_wise(eeg)

    eeg_norm = normalize_channel_wise(eeg)

    MAX_CH = 128
    MAX_TIME = 1000

    C, T = eeg_norm.shape

    if C < MAX_CH:
        pad_c = np.zeros((MAX_CH - C, T), dtype=eeg_norm.dtype)
        eeg_fixed = np.vstack([eeg_norm, pad_c])
    else:
        eeg_fixed = eeg_norm[:MAX_CH, :]

    processed_C, processed_T = eeg_fixed.shape
    if processed_T < MAX_TIME:
        pad_t = np.zeros((processed_C, MAX_TIME - processed_T), dtype=eeg_fixed.dtype)
        eeg_fixed = np.hstack([eeg_fixed, pad_t])
    else:
        eeg_fixed = eeg_fixed[:, :MAX_TIME]

    specs = generate_spectrogram(eeg_fixed, fs)
    return torch.tensor(eeg_fixed, dtype=torch.float32), torch.tensor(specs, dtype=torch.float32)
