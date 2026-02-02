import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.transformer_model import EEGTransformer
from src.train import train_epoch, validate
# from src.evaluate import evaluate_model # Evaluate might need update too

# Config
DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"
BATCH_SIZE = 4 # Small batch logic
EPOCHS = 1 
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Dataset Discovery
print("Scanning dataset...")
files, labels, subjects = get_all_file_paths(DATA_ROOT)
files = np.array(files)
labels = np.array(labels)
subjects = np.array(subjects)
unique_subjects = np.unique(subjects)

print(f"Total samples: {len(files)}")
print(f"Subjects found: {unique_subjects}")

# 2. LOSO Loop
results = {}

# Demo: only run 1 subject
for test_subject in unique_subjects[:1]:
    print(f"\n{'='*30}")
    print(f"LOSO Fold: Testing on {test_subject}")
    print(f"{'='*30}")
    
    # Split
    test_mask = (subjects == test_subject)
    train_mask = ~test_mask
    
    X_train, y_train = files[train_mask], labels[train_mask]
    X_test, y_test = files[test_mask], labels[test_mask]
    
    if len(X_test) == 0:
        continue

    print("[INFO] Subsampling for Demo...")
    # Shuffle
    perm_train = np.random.permutation(len(X_train))[:100] # 100 samples
    X_train = X_train[perm_train]
    y_train = y_train[perm_train]
    
    perm_test = np.random.permutation(len(X_test))[:20]
    X_test = X_test[perm_test]
    y_test = y_test[perm_test]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Check class balance
    if len(np.unique(y_train)) < 2:
        print("Warning: Train set has only 1 class! Skipping.")
        continue
    
    # Compute Weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class Weights: {weight_tensor}")
    
    # Loaders
    train_ds = EEGDataset(X_train, y_train)
    test_ds = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = EEGTransformer(num_channels=MAX_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # Train
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
