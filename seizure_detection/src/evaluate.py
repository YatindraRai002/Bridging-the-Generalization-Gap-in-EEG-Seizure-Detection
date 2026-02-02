from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for raw, spec, labels in loader:
            raw = raw.to(device)
            spec = spec.to(device)
            
            outputs = model(raw, spec)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=['Interictal', 'Ictal'])
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    return all_labels, all_preds
