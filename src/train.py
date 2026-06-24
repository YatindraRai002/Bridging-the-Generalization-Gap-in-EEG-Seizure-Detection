import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

def train_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for data, labels in pbar:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return total_loss / len(loader), accuracy, sensitivity, specificity

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return total_loss / len(loader), accuracy, sensitivity, specificity, np.array(all_preds), np.array(all_labels)

