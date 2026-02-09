
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix

from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.baseline_cnn import BaselineCNN
from src.train import train_epoch, validate

DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"
BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-5

# Patient Split
TRAIN_SUBJECTS = ['Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6']
VAL_SUBJECT = 'Patient_8'
TEST_SUBJECT = 'Patient_7'

PATIENCE = 15 # Shorter patience for validation
MIN_DELTA = 0.001

USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

OUTPUT_PREFIX = "experiment_P7_P8"

def run_custom_experiment():
    if not torch.cuda.is_available():
        print("[!] WARNING: CUDA NOT AVAILABLE. Training will be slow on CPU!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable AMP
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    print(f"[*] Using device: {device}")
    print(f"[*] AMP Enabled: {use_amp}")
    
    # Optimize CUDNN
    torch.backends.cudnn.benchmark = True
    print(f"[*] CUDNN Benchmark Enabled")
    
    # 1. Dataset Discovery
    print("\n[*] Scanning dataset...")
    files, labels, subjects = get_all_file_paths(DATA_ROOT)
    files = np.array(files)
    labels = np.array(labels)
    subjects = np.array(subjects)
    
    print(f"   Total samples in root: {len(files)}")
    
    # 2. Split Data
    train_mask = np.isin(subjects, TRAIN_SUBJECTS)
    val_mask = (subjects == VAL_SUBJECT)
    test_mask = (subjects == TEST_SUBJECT)
    
    X_train, y_train = files[train_mask], labels[train_mask]
    X_val, y_val = files[val_mask], labels[val_mask]
    X_test, y_test = files[test_mask], labels[test_mask]
    
    print(f"[*] Train size: {len(X_train)} (Subjects: {TRAIN_SUBJECTS})")
    print(f"[*] Val size:   {len(X_val)} (Subject: {VAL_SUBJECT})")
    print(f"[*] Test size:  {len(X_test)} (Subject: {TEST_SUBJECT})")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("[X] Error: One of the splits is empty. Please check subject names.")
        return

    # 3. Data Loaders
    # Weighted Sampler for Train split
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for t in y_train])
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_ds = EEGDataset(X_train, y_train)
    val_ds = EEGDataset(X_val, y_val)
    test_ds = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    # 4. Model & Optimization
    model = BaselineCNN(num_channels=MAX_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
        )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    best_val_loss = float('inf')
    history_log = []
    start_epoch = 0
    trigger_times = 0

    # Resume logic
    history_file = f"{OUTPUT_PREFIX}_history.csv"
    model_file = f"{OUTPUT_PREFIX}_best_model.pth"

    if os.path.exists(history_file):
        print(f"[*] Found history file {history_file}. Loading...")
        df_history = pd.read_csv(history_file)
        history_log = df_history.to_dict('records')
        if len(history_log) > 0:
            start_epoch = history_log[-1]['Epoch']
            best_val_loss = df_history['Val_Loss'].min()
            print(f"[*] Resuming from epoch {start_epoch}. Best Val Loss so far: {best_val_loss:.4f}")

    if os.path.exists(model_file):
        print(f"[*] Found best model {model_file}. Loading weights...")
        model.load_state_dict(torch.load(model_file, map_location=device))
    
    print(f"\n{'='*120}")
    print(f"{'Epoch':<6} | {'Tr Loss':<9} {'Tr Acc':<9} {'Tr Sens':<9} {'Tr Spec':<9} | {'Val Loss':<9} {'Val Acc':<9} {'Val Sens':<9} {'Val Spec':<9} | {'Val F1':<9} | {'Status'}")
    print(f"{'='*120}")
    
    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc, train_sens, train_spec = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler=scaler)
        val_loss, val_acc, val_sens, val_spec, val_preds, val_labels = validate(model, val_loader, loss_fn, device)
        
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        # Log entry
        entry = {
            'Epoch': epoch + 1,
            'Train_Loss': train_loss, 'Train_Acc': train_acc, 'Train_Sens': train_sens, 'Train_Spec': train_spec,
            'Val_Loss': val_loss, 'Val_Acc': val_acc, 'Val_Sens': val_sens, 'Val_Spec': val_spec, 'Val_F1': val_f1,
            'LR': optimizer.param_groups[0]['lr']
        }
        history_log.append(entry)
        pd.DataFrame(history_log).to_csv(f"{OUTPUT_PREFIX}_history.csv", index=False)
        
        # Best model tracking
        status = ""
        if val_loss < best_val_loss - MIN_DELTA:
            status = "[BEST]"
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), f"{OUTPUT_PREFIX}_best_model.pth")
        else:
            trigger_times += 1
            status = f"[Wait {trigger_times}/{PATIENCE}]"
            
        print(f"{epoch+1:<6} | {train_loss:<9.4f} {train_acc:<9.4f} {train_sens:<9.4f} {train_spec:<9.4f} | "
              f"{val_loss:<9.4f} {val_acc:<9.4f} {val_sens:<9.4f} {val_spec:<9.4f} | "
              f"{val_f1:<9.4f} | {status}")
        
        if USE_LR_SCHEDULER:
            scheduler.step(val_loss)
            
        if trigger_times >= PATIENCE:
            print(f"\n[STOP] Early stopping at epoch {epoch+1}")
            break
            
    # 6. Final Evaluation on TEST SUBJECT
    print(f"\n{'='*60}")
    print(f"[*] Final Testing on {TEST_SUBJECT}")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load(f"{OUTPUT_PREFIX}_best_model.pth", map_location=device))
    test_loss, test_acc, test_sens, test_spec, test_preds, test_labels = validate(model, test_loader, loss_fn, device)
    
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    print(f"   Test Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Test F1 Score:    {test_f1:.4f}")
    print(f"   Test Sensitivity: {test_sens:.4f}")
    print(f"   Test Specificity: {test_spec:.4f}")
    print(f"\n   Confusion Matrix:\n   TN: {tn:4d}  FP: {fp:4d}\n   FN: {fn:4d}  TP: {tp:4d}")
    
    # Save Final Results
    final_results = [{
        'Experiment': 'Train 1-6, Val 8, Test 7',
        'Accuracy': test_acc,
        'F1': test_f1,
        'Sensitivity': test_sens,
        'Specificity': test_spec,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }]
    pd.DataFrame(final_results).to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)
    print(f"\n[DONE] Experiment complete. Results saved to {OUTPUT_PREFIX}_results.csv")

if __name__ == "__main__":
    run_custom_experiment()
