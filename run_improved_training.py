"""
Improved Training Script with Extended Epochs and Enhanced Monitoring
Based on analysis showing model is actively learning and needs more training time
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.baseline_cnn import BaselineCNN
from src.train import train_epoch, validate

DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"
BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-5

PATIENCE = 50
MIN_DELTA = 0.001

USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

OUTPUT_FILE = "loso_results_improved.csv"

def run_improved_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    print(f"[*] Training Configuration:")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Learning Rate: {LR}")
    print(f"   - Early Stopping Patience: {PATIENCE}")
    print(f"   - LR Scheduler: {'Enabled' if USE_LR_SCHEDULER else 'Disabled'}")

    print("\n[*] Scanning dataset...")
    files, labels, subjects = get_all_file_paths(DATA_ROOT)
    files = np.array(files)
    labels = np.array(labels)
    subjects = np.array(subjects)
    unique_subjects = np.unique(subjects)

    print(f"   Total samples: {len(files)}")
    print(f"   Subjects found: {len(unique_subjects)}")

    results = []

    target_subjects = ['Patient_4', 'Patient_5', 'Patient_6']

    for test_subject in target_subjects:

        history_log = []
        if test_subject not in unique_subjects:
            print(f"[X] Subject {test_subject} not found in dataset!")
            continue

        print(f"\n{'='*60}")
        print(f"[*] LOSO Fold: Testing on {test_subject}")
        print(f"{'='*60}")

        test_mask = (subjects == test_subject)
        train_mask = ~test_mask

        X_train, y_train = files[train_mask], labels[train_mask]
        X_test, y_test = files[test_mask], labels[test_mask]

        if len(X_test) == 0:
            print("[!] Skipping empty test subject...")
            continue

        print(f"[*] Train size: {len(X_train)} | Test size: {len(X_test)}")

        unique, counts = np.unique(y_train, return_counts=True)
        train_dist = dict(zip(unique, counts))
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        test_dist = dict(zip(unique_test, counts_test))

        print(f"[+] Train Class Distribution: {train_dist}")
        print(f"[-] Test Class Distribution: {test_dist}")

        if len(train_dist) == 2:
            imbalance_ratio = max(train_dist.values()) / min(train_dist.values())
            print(f"[=] Class Imbalance Ratio: {imbalance_ratio:.2f}:1")

        if len(np.unique(y_train)) < 2:
            print("[!] Warning: Train set has only 1 class! Skipping.")
            continue

        class_counts = np.bincount(y_train)
        class_weights = 1. / class_counts
        sample_weights = np.array([class_weights[t] for t in y_train])

        print(f"[=] Class Weights: {class_weights}")

        sample_weights = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_ds = EEGDataset(X_train, y_train)
        test_ds = EEGDataset(X_test, y_test)

        sample_x, sample_y = train_ds[0]
        print(f"[*] Model Input Shape: {sample_x.shape}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        model = BaselineCNN(num_channels=MAX_CHANNELS).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

        if USE_LR_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=LR_SCHEDULER_FACTOR,
                patience=LR_SCHEDULER_PATIENCE
            )
            print(f"[*] Learning Rate Scheduler enabled (patience={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR})")

        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_val_f1 = 0.0
        trigger_times = 0

        print(f"\n{'='*120}")
        print(f"{'Epoch':<6} | {'Tr Loss':<9} {'Tr Acc':<9} {'Tr Sens':<9} {'Tr Spec':<9} | {'Val Loss':<9} {'Val Acc':<9} {'Val Sens':<9} {'Val Spec':<9} | {'Val F1':<9} | {'Gap':<9} {'Status'}")
        print(f"{'='*120}")

        for epoch in range(EPOCHS):
            train_loss, train_acc, train_sens, train_spec = train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc, val_sens, val_spec, val_preds, val_labels = validate(model, test_loader, loss_fn, device)

            val_f1 = f1_score(val_labels, val_preds, zero_division=0)

            acc_gap = train_acc - val_acc

            history_log.append({
                'Subject': test_subject,
                'Epoch': epoch + 1,
                'Train_Loss': train_loss,
                'Train_Acc': train_acc,
                'Train_Sens': train_sens,
                'Train_Spec': train_spec,
                'Val_Loss': val_loss,
                'Val_Acc': val_acc,
                'Val_Sens': val_sens,
                'Val_Spec': val_spec,
                'Val_F1': val_f1,
                'Acc_Gap': acc_gap,
                'LR': optimizer.param_groups[0]['lr']
            })

            pd.DataFrame(history_log).to_csv(f"history_{test_subject}_improved.csv", index=False)

            status = ""
            if val_loss < best_val_loss - MIN_DELTA:
                status = "[BEST]"
                best_val_loss = val_loss
                trigger_times = 0
                torch.save(model.state_dict(), f"model_{test_subject}_improved.pth")
            else:
                trigger_times += 1
                status = f"[Wait {trigger_times}/{PATIENCE}]"

            gap_status = ""
            if acc_gap > 0.15:
                gap_status = "[High Gap]"
            elif acc_gap > 0.10:
                gap_status = "[Med Gap]"
            else:
                gap_status = "[Good]"

            print(f"{epoch+1:<6} | {train_loss:<9.4f} {train_acc:<9.4f} {train_sens:<9.4f} {train_spec:<9.4f} | "
                  f"{val_loss:<9.4f} {val_acc:<9.4f} {val_sens:<9.4f} {val_spec:<9.4f} | "
                  f"{val_f1:<9.4f} | {acc_gap:<9.4f} {status} {gap_status}")

            if USE_LR_SCHEDULER:
                scheduler.step(val_loss)

            if trigger_times >= PATIENCE:
                print(f"\n[STOP] Early stopping triggered at epoch {epoch+1}")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                break

        print(f"{'='*120}\n")

        print("[*] Loading best model for final evaluation...")
        model.load_state_dict(torch.load(f"model_{test_subject}_improved.pth", map_location=device))

        test_loss, acc, sens, spec, all_preds, all_labels = validate(model, test_loader, loss_fn, device)

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"\n{'='*60}")
        print(f"[RESULTS] Final Results for {test_subject}")
        print(f"{'='*60}")
        print(f"   Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
        print(f"   F1 Score:    {f1:.4f}")
        print(f"   Sensitivity: {sens:.4f} ({sens*100:.2f}%)")
        print(f"   Specificity: {spec:.4f} ({spec*100:.2f}%)")
        print(f"\n   Confusion Matrix:")
        print(f"   TN: {tn:4d}  FP: {fp:4d}")
        print(f"   FN: {fn:4d}  TP: {tp:4d}")
        print(f"{'='*60}\n")

        results.append({
            'Subject': test_subject,
            'Accuracy': acc,
            'F1': f1,
            'Sensitivity': sens,
            'Specificity': spec,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        })

        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    print(f"\n[DONE] Training complete! Results saved to {OUTPUT_FILE}")
    print(f"[*] History saved to history_{test_subject}_improved.csv")

if __name__ == "__main__":
    run_improved_training()
