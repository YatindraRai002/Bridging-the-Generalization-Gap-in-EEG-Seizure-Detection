import torch
import glob
import os
import numpy as np
from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.transformer_model import EEGTransformer
from src.explainability import visualize_channel_attention

DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"

def run_explainability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Find trained models
    model_files = glob.glob("model_*.pth")
    if not model_files:
        print("No trained models found. Run experiments first.")
        return

    print(f"Found {len(model_files)} models.")
    
    # Pick the first one for demo
    model_path = model_files[0]
    subject = model_path.replace("model_", "").replace(".pth", "")
    print(f"Analyzing Model for Subject: {subject}")
    
    # Load Data for this subject (just one sample)
    # Ideally find an Ictal sample to see what triggered detection
    print("Loading one ictal sample...")
    
    # Quick dirty search for an ictal file for this subject
    # Assuming standard path structure
    subject_dir = os.path.join(DATA_ROOT, subject)
    ictal_files = glob.glob(os.path.join(subject_dir, "*ictal*.mat"))
    ictal_files = [f for f in ictal_files if 'interictal' not in f]
    
    if not ictal_files:
        print("No ictal files found for this subject.")
        return
        
    # Load dataset wrapper just for this file
    ds = EEGDataset([ictal_files[0]], [1])
    data, label = ds[0] # (C, T), scalar
    
    # Add batch dim
    data = data.unsqueeze(0) # (1, C, T)
    
    # Load Model
    model = EEGTransformer(num_channels=MAX_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Visualise
    importances = visualize_channel_attention(model, data, device=device)
    
    # Top 5 channels
    top_indices = np.argsort(importances)[::-1][:5]
    print("Top 5 Contributing Channels (Indices):", top_indices)
    print("Note: Correlate these indices with physical electrode locations if montage is known.")

if __name__ == "__main__":
    run_explainability()
