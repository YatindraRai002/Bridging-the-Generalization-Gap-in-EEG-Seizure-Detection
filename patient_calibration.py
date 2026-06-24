
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.baseline_cnn import BaselineCNN
from src.train import validate, train_epoch

DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"
BASE_MODEL_PATH = "seizure_detection/base_model_independent.pth"
ADAPT_SUBJECT = 'Patient_7'
BATCH_SIZE = 64
CALIBRATION_FRACTION = 0.2
EPOCHS_ADAPT = 10
LR_ADAPT = 1e-4

def run_patient_adaptation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Calibration Device: {device}")

    print(f"[*] Scanning data for {ADAPT_SUBJECT}...")
    files, labels, subjects = get_all_file_paths(DATA_ROOT)
    mask = (np.array(subjects) == ADAPT_SUBJECT)
    X_subject = np.array(files)[mask]
    y_subject = np.array(labels)[mask]

    print(f"[*] Total samples for {ADAPT_SUBJECT}: {len(X_subject)}")

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_subject, y_subject, test_size=(1 - CALIBRATION_FRACTION), stratify=y_subject, random_state=42
    )

    print(f"[*] Calibration size: {len(X_cal)}")
    print(f"[*] Test size:        {len(X_test)}")

    cal_ds = EEGDataset(X_cal, y_cal)
    test_ds = EEGDataset(X_test, y_test)

    cal_loader = DataLoader(cal_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[*] Evaluating Baseline Performance (Frozen Base Model)...")
    model = BaselineCNN(num_channels=MAX_CHANNELS).to(device)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))

    loss_fn = nn.CrossEntropyLoss()
    _, b_acc, b_sens, b_spec, b_preds, b_labels = validate(model, test_loader, loss_fn, device)
    b_f1 = f1_score(b_labels, b_preds, zero_division=0)

    print(f"   Baseline Acc:  {b_acc:.4f} | Sens: {b_sens:.4f} | Spec: {b_spec:.4f} | F1: {b_f1:.4f}")

    print("\n[*] Mode A: Decision Threshold Optimization...")

    model.eval()
    all_probs = []
    with torch.no_grad():
        for signals, _ in cal_loader:
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    best_threshold = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (np.array(all_probs) >= t).astype(int)
        f1 = f1_score(y_cal, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"   Optimal Threshold found: {best_threshold:.3f} (F1: {best_f1:.4f})")

    test_probs = []
    with torch.no_grad():
        for signals, _ in test_loader:
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)

    t_preds = (np.array(test_probs) >= best_threshold).astype(int)
    t_acc = np.mean(t_preds == y_test)
    t_sens = np.sum((t_preds == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    t_spec = np.sum((t_preds == 0) & (y_test == 0)) / np.sum(y_test == 0) if np.sum(y_test == 0) > 0 else 0
    t_f1 = f1_score(y_test, t_preds, zero_division=0)

    print(f"   Threshold Adaptation results on Test:")
    print(f"   Acc: {t_acc:.4f} | Sens: {t_sens:.4f} | Spec: {t_spec:.4f} | F1: {t_f1:.4f}")

    print("\n[*] Mode B: Fine-tuning Head (CNN Frozen)...")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR_ADAPT)

    for epoch in range(EPOCHS_ADAPT):
        train_epoch(model, cal_loader, optimizer, loss_fn, device)

    _, h_acc, h_sens, h_spec, h_preds, h_labels = validate(model, test_loader, loss_fn, device)
    h_f1 = f1_score(h_labels, h_preds, zero_division=0)

    print(f"   Final Head-Tuned results on Test:")
    print(f"   Acc: {h_acc:.4f} | Sens: {h_sens:.4f} | Spec: {h_spec:.4f} | F1: {h_f1:.4f}")

    results = pd.DataFrame([{
        'Patient': ADAPT_SUBJECT,
        'Base_Sens': b_sens, 'Base_Spec': b_spec, 'Base_F1': b_f1,
        'Threshold_Sens': t_sens, 'Threshold_Spec': t_spec, 'Threshold_F1': t_f1, 'Best_Threshold': best_threshold,
        'Head_Sens': h_sens, 'Head_Spec': h_spec, 'Head_F1': h_f1
    }])
    results.to_csv(f"seizure_detection/calibration_results_{ADAPT_SUBJECT}.csv", index=False)
    print(f"\n[DONE] Saved results to seizure_detection/calibration_results_{ADAPT_SUBJECT}.csv")

if __name__ == "__main__":
    run_patient_adaptation()
