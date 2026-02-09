
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.baseline_cnn import BaselineCNN


DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"
BASE_MODEL_PATH = "seizure_detection/base_model_independent.pth"
SUBJECTS_TO_ANALYZE = ['Patient_7', 'Patient_8', 'Patient_1']
SAMPLES_PER_SUBJECT = 200
BATCH_SIZE = 64

def extract_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Extraction Device: {device}")
    
    model = BaselineCNN(num_channels=MAX_CHANNELS).to(device)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    model.eval()
    
    def get_embedding(x):
        features = model.features(x)
        x = model.classifier[0](features) 
        x = model.classifier[1](x) 
        x = model.classifier[2](x) 
        return x

    # 2. Collect Data
    print("[*] Scanning dataset...")
    files, labels, subjects = get_all_file_paths(DATA_ROOT)
    files = np.array(files)
    labels = np.array(labels)
    subjects = np.array(subjects)
    
    all_embeddings = []
    all_labels = []
    all_subject_ids = []
    
    for sub in SUBJECTS_TO_ANALYZE:
        mask = (subjects == sub)
        sub_files = files[mask]
        sub_labels = labels[mask]
        
        # Sample subset
        idx = np.random.choice(len(sub_files), min(len(sub_files), SAMPLES_PER_SUBJECT), replace=False)
        sample_files = sub_files[idx]
        sample_labels = sub_labels[idx]
        
        ds = EEGDataset(sample_files, sample_labels)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"[*] Extracting embeddings for {sub}...")
        with torch.no_grad():
            for signals, batch_labels in loader: # Using batch_labels from loader
                signals = signals.to(device)
                emb = get_embedding(signals)
                all_embeddings.append(emb.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_subject_ids.extend([sub] * len(signals))
                
    X_emb = np.concatenate(all_embeddings, axis=0)
    y_lbl = np.array(all_labels)
    y_sub = np.array(all_subject_ids)
    
    # 3. TSNE Projection
    print(f"[*] Total samples for t-SNE: {len(X_emb)}")
    print("[*] Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_emb)//3), random_state=42)
    X_2d = tsne.fit_transform(X_emb)
    
    # 4. Visualization
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Color by Subject
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_sub, palette='viridis', alpha=0.7)
    plt.title('Embeddings by Patient ID')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Subplot 2: Color by Seizure Label
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_lbl, palette='coolwarm', alpha=0.7)
    plt.title('Embeddings by Seizure Label')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/embedding_analysis_tsne.png', dpi=300)
    print("[OK] Saved: images/embedding_analysis_tsne.png")

if __name__ == "__main__":
    extract_embeddings()
