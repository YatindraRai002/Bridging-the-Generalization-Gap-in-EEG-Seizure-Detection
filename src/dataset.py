import os
import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
from src.preprocess import preprocess_pipeline, TARGET_FREQ

MAX_CHANNELS = 72

FIXED_SAMPLES = int(1.0 * TARGET_FREQ)

class EEGDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): List of absolute paths.
            labels (list): List of int labels.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:

            mat = scipy.io.loadmat(path)
            if 'data' in mat:
                data = mat['data']
                if 'sampling_frequency' in mat:
                    sf = mat['sampling_frequency'].flat[0]
                elif 'freq' in mat:
                    sf = mat['freq'].flat[0]
                else:

                    sf = 5000
            else:

                 keys = [k for k in mat.keys() if not k.startswith('__')]
                 data = mat[keys[0]]['data'][0,0]
                 sf = mat[keys[0]]['sampling_frequency'][0,0][0,0]

            if data.shape[0] > data.shape[1]:

                 pass

            if data.shape[0] > data.shape[1]:

                 data = data.T

            data = preprocess_pipeline(data, sf)

            C, T = data.shape

            if T > FIXED_SAMPLES:
                data = data[:, :FIXED_SAMPLES]
            elif T < FIXED_SAMPLES:
                 pad = np.zeros((C, FIXED_SAMPLES - T))
                 data = np.concatenate([data, pad], axis=1)

            if C < MAX_CHANNELS:
                pad_c = np.zeros((MAX_CHANNELS - C, FIXED_SAMPLES))
                data = np.concatenate([data, pad_c], axis=0)
            elif C > MAX_CHANNELS:
                data = data[:MAX_CHANNELS, :]

            data_tensor = torch.tensor(data, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return data_tensor, label_tensor

        except Exception as e:

            print(f"Error loading {path}: {e}")
            return torch.zeros((MAX_CHANNELS, FIXED_SAMPLES)), torch.tensor(0)

def get_all_file_paths(root_dir):
    all_files = []
    all_labels = []
    subjects = []

    if not os.path.exists(root_dir):
         raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    for item in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, item)
        if os.path.isdir(patient_path):
            files = [f for f in os.listdir(patient_path) if f.endswith('.mat')]
            for fname in files:
                if 'test' in fname: continue

                label = 1 if 'ictal' in fname and 'interictal' not in fname else 0

                all_files.append(os.path.join(patient_path, fname))
                all_labels.append(label)
                subjects.append(item)

    return all_files, all_labels, subjects
