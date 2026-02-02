
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc

from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.baseline_cnn import BaselineCNN

# Configuration
DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"
MODEL_PATH = "model_Patient_1_medical.pth"
TEST_SUBJECT = "Patient_1"
BATCH_SIZE = 64

# Test thresholds from 0.20 to 0.55
THRESHOLDS = np.arange(0.20, 0.56, 0.05)

def evaluate_at_threshold(model, loader, device, threshold):
    """Evaluate model at specific threshold"""
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            outputs = model(data)
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # Get probabilities
    probs = torch.softmax(all_outputs, dim=1)
    seizure_probs = probs[:, 1].cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    
    # Apply threshold
    preds = (seizure_probs >= threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(labels_np, preds, zero_division=0)
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'probs': seizure_probs,
        'labels': labels_np
    }

def find_optimal_threshold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"🎯 THRESHOLD OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test Subject: {TEST_SUBJECT}")
    print(f"Testing {len(THRESHOLDS)} thresholds: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f}")
    
    # Load dataset
    print(f"\n📁 Loading dataset...")
    files, labels, subjects = get_all_file_paths(DATA_ROOT)
    files = np.array(files)
    labels = np.array(labels)
    subjects = np.array(subjects)
    
    # Get test subject data
    test_mask = (subjects == TEST_SUBJECT)
    X_test, y_test = files[test_mask], labels[test_mask]
    
    print(f"   Test samples: {len(X_test)}")
    
    # Create test loader
    test_ds = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load model
    print(f"\n🧠 Loading model...")
    model = BaselineCNN(num_channels=MAX_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Evaluate at each threshold
    print(f"\n{'='*120}")
    print(f"{'Threshold':<12} | {'Sens':<8} {'Spec':<8} {'Prec':<8} {'F1':<8} {'Acc':<8} | {'TP':<5} {'TN':<5} {'FP':<5} {'FN':<5} | {'Status'}")
    print(f"{'='*120}")
    
    results = []
    best_f1 = 0
    best_balanced = 0
    best_threshold_f1 = 0
    best_threshold_balanced = 0
    
    for threshold in THRESHOLDS:
        metrics = evaluate_at_threshold(model, test_loader, device, threshold)
        results.append(metrics)
        
        # Track best F1
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold_f1 = threshold
        
        # Track best balanced (sensitivity + specificity)
        balance_score = metrics['sensitivity'] + metrics['specificity']
        if balance_score > best_balanced:
            best_balanced = balance_score
            best_threshold_balanced = threshold
        
        # Status indicators
        sens_status = "🟢" if metrics['sensitivity'] >= 0.70 else "🟡" if metrics['sensitivity'] >= 0.50 else "🔴"
        spec_status = "🟢" if metrics['specificity'] >= 0.60 else "🟡" if metrics['specificity'] >= 0.40 else "🔴"
        
        status = ""
        if metrics['sensitivity'] >= 0.70 and metrics['specificity'] >= 0.60:
            status = "✅ OPTIMAL"
        elif metrics['sensitivity'] >= 0.60 and metrics['specificity'] >= 0.50:
            status = "🟡 GOOD"
        
        print(f"{threshold:<12.2f} | {metrics['sensitivity']:<8.3f}{sens_status} {metrics['specificity']:<8.3f}{spec_status} "
              f"{metrics['precision']:<8.3f} {metrics['f1']:<8.3f} {metrics['accuracy']:<8.3f} | "
              f"{metrics['tp']:<5d} {metrics['tn']:<5d} {metrics['fp']:<5d} {metrics['fn']:<5d} | {status}")
    
    print(f"{'='*120}\n")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("threshold_optimization_results.csv", index=False)
    print(f"✅ Results saved to threshold_optimization_results.csv")
    
    # Recommendations
    print(f"\n{'='*80}")
    print(f"🎯 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"\n1️⃣ Best F1 Score:")
    best_f1_metrics = results[np.argmax([r['f1'] for r in results])]
    print(f"   Threshold: {best_f1_metrics['threshold']:.2f}")
    print(f"   Sensitivity: {best_f1_metrics['sensitivity']:.3f} ({best_f1_metrics['sensitivity']*100:.1f}%)")
    print(f"   Specificity: {best_f1_metrics['specificity']:.3f} ({best_f1_metrics['specificity']*100:.1f}%)")
    print(f"   F1 Score: {best_f1_metrics['f1']:.3f}")
    
    print(f"\n2️⃣ Best Balanced (Sensitivity + Specificity):")
    best_bal_metrics = results[np.argmax([r['sensitivity'] + r['specificity'] for r in results])]
    print(f"   Threshold: {best_bal_metrics['threshold']:.2f}")
    print(f"   Sensitivity: {best_bal_metrics['sensitivity']:.3f} ({best_bal_metrics['sensitivity']*100:.1f}%)")
    print(f"   Specificity: {best_bal_metrics['specificity']:.3f} ({best_bal_metrics['specificity']*100:.1f}%)")
    print(f"   F1 Score: {best_bal_metrics['f1']:.3f}")
    
    # Find medical-grade threshold (Sens >= 70%, Spec >= 60%)
    medical_grade = [r for r in results if r['sensitivity'] >= 0.70 and r['specificity'] >= 0.60]
    if medical_grade:
        print(f"\n3️⃣ Medical-Grade Threshold (Sens≥70%, Spec≥60%):")
        best_medical = max(medical_grade, key=lambda x: x['f1'])
        print(f"   Threshold: {best_medical['threshold']:.2f} ✅")
        print(f"   Sensitivity: {best_medical['sensitivity']:.3f} ({best_medical['sensitivity']*100:.1f}%)")
        print(f"   Specificity: {best_medical['specificity']:.3f} ({best_medical['specificity']*100:.1f}%)")
        print(f"   F1 Score: {best_medical['f1']:.3f}")
    else:
        print(f"\n3️⃣ Medical-Grade Threshold:")
        print(f"   ⚠️ No threshold meets Sens≥70% AND Spec≥60%")
        print(f"   Consider: Retrain with adjusted weights or accept lower specificity")
    
    # Create visualizations
    create_threshold_plots(results)
    
    print(f"\n{'='*80}\n")
    return results

def create_threshold_plots(results):
    """Create threshold optimization visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Threshold Optimization Analysis', fontsize=16, fontweight='bold')
    
    thresholds = [r['threshold'] for r in results]
    
    # 1. Sensitivity vs Specificity
    ax1 = axes[0, 0]
    ax1.plot(thresholds, [r['sensitivity'] for r in results], 'o-', label='Sensitivity', linewidth=2, markersize=8, color='#2ecc71')
    ax1.plot(thresholds, [r['specificity'] for r in results], 's-', label='Specificity', linewidth=2, markersize=8, color='#e74c3c')
    ax1.axhline(y=0.70, color='green', linestyle='--', alpha=0.3, label='Target Sens (70%)')
    ax1.axhline(y=0.60, color='red', linestyle='--', alpha=0.3, label='Target Spec (60%)')
    ax1.set_xlabel('Decision Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Sensitivity vs Specificity', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score
    ax2 = axes[0, 1]
    ax2.plot(thresholds, [r['f1'] for r in results], 'o-', linewidth=2, markersize=8, color='#9b59b6')
    best_f1_idx = np.argmax([r['f1'] for r in results])
    ax2.axvline(x=thresholds[best_f1_idx], color='orange', linestyle='--', alpha=0.5, label=f'Best F1 @ {thresholds[best_f1_idx]:.2f}')
    ax2.set_xlabel('Decision Threshold', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Tradeoff
    ax3 = axes[1, 0]
    ax3.plot([r['sensitivity'] for r in results], [r['precision'] for r in results], 'o-', linewidth=2, markersize=8, color='#3498db')
    for i, thresh in enumerate(thresholds[::2]):  # Label every other point
        idx = i * 2
        ax3.annotate(f'{thresh:.2f}', (results[idx]['sensitivity'], results[idx]['precision']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    ax3.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy
    ax4 = axes[1, 1]
    ax4.plot(thresholds, [r['accuracy'] for r in results], 'o-', linewidth=2, markersize=8, color='#f39c12')
    ax4.set_xlabel('Decision Threshold', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_optimization_plots.png', dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to threshold_optimization_plots.png")
    plt.close()

if __name__ == "__main__":
    results = find_optimal_threshold()
