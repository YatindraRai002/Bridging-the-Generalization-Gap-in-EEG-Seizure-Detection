import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

df = pd.read_csv('history_Patient_1_improved.csv')

print(f"Loaded {len(df)} epochs of training data")
print(f"\nFinal Metrics:")
print(f"  Validation Accuracy: {df['Val_Acc'].iloc[-1]:.4f}")
print(f"  Validation Sensitivity: {df['Val_Sens'].iloc[-1]:.4f}")
print(f"  Validation Specificity: {df['Val_Spec'].iloc[-1]:.4f}")
print(f"  Validation F1: {df['Val_F1'].iloc[-1]:.4f}")

fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(3, 3, 1)
plt.plot(df['Epoch'], df['Train_Loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
plt.plot(df['Epoch'], df['Val_Loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.title('Loss Curve', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
plt.plot(df['Epoch'], df['Train_Acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
plt.plot(df['Epoch'], df['Val_Acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Accuracy Curve', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
plt.plot(df['Epoch'], df['Val_F1'], 'g-o', label='Val F1', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('F1 Score', fontweight='bold')
plt.title('F1 Score Progression', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 3, 4)
plt.plot(df['Epoch'], df['Train_Sens'], 'b-o', label='Train Sensitivity', linewidth=2, markersize=4)
plt.plot(df['Epoch'], df['Val_Sens'], 'r-s', label='Val Sensitivity', linewidth=2, markersize=4)
plt.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target (85%)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Sensitivity (Recall)', fontweight='bold')
plt.title('Sensitivity - Seizure Detection Rate', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
plt.plot(df['Epoch'], df['Train_Spec'], 'b-o', label='Train Specificity', linewidth=2, markersize=4)
plt.plot(df['Epoch'], df['Val_Spec'], 'r-s', label='Val Specificity', linewidth=2, markersize=4)
plt.axhline(y=0.50, color='green', linestyle='--', alpha=0.5, label='Target (50%)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Specificity', fontweight='bold')
plt.title('Specificity - Non-Seizure Detection Rate', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
plt.plot(df['Epoch'], df['Acc_Gap'], 'purple', linewidth=2, marker='o', markersize=4)
plt.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Warning Threshold (15%)')
plt.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Caution (10%)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Gap (Train - Val Acc)', fontweight='bold')
plt.title('Overfitting Gap Analysis', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

ax7 = plt.subplot(3, 3, 7)
plt.plot(df['Epoch'], df['LR'], 'orange', linewidth=2, marker='o', markersize=4)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Learning Rate', fontweight='bold')
plt.title('Learning Rate Schedule', fontweight='bold', fontsize=12)
plt.yscale('log')
plt.grid(True, alpha=0.3)

ax8 = plt.subplot(3, 3, 8)
plt.plot(df['Val_Spec'], df['Val_Sens'], 'bo-', linewidth=2, markersize=6)

for i in [0, len(df)//2, len(df)-1]:
    plt.annotate(f'E{df["Epoch"].iloc[i]}',
                xy=(df['Val_Spec'].iloc[i], df['Val_Sens'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.xlabel('Specificity', fontweight='bold')
plt.ylabel('Sensitivity', fontweight='bold')
plt.title('Sensitivity-Specificity Trade-off', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_data = [
    ['Metric', 'Best Epoch', 'Final Epoch'],
    ['Val Accuracy', f"{df['Val_Acc'].max():.4f}", f"{df['Val_Acc'].iloc[-1]:.4f}"],
    ['Val Sensitivity', f"{df['Val_Sens'].max():.4f}", f"{df['Val_Sens'].iloc[-1]:.4f}"],
    ['Val Specificity', f"{df['Val_Spec'].max():.4f}", f"{df['Val_Spec'].iloc[-1]:.4f}"],
    ['Val F1 Score', f"{df['Val_F1'].max():.4f}", f"{df['Val_F1'].iloc[-1]:.4f}"],
    ['Overfitting Gap', f"{df['Acc_Gap'].min():.4f}", f"{df['Acc_Gap'].iloc[-1]:.4f}"],
    ['Learning Rate', f"{df['LR'].max():.2e}", f"{df['LR'].iloc[-1]:.2e}"],
]

table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(summary_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

plt.suptitle('EEG Seizure Detection - Training Analysis (Patient_1, 16 Epochs)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('images/training_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: images/training_comprehensive_analysis.png")

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df['Epoch'], df['Val_Sens'], 'r-o', label='Sensitivity (Seizure Detection)', linewidth=2.5)
axes[0, 0].plot(df['Epoch'], df['Val_Spec'], 'b-s', label='Specificity (Non-Seizure Detection)', linewidth=2.5)
axes[0, 0].axhline(y=0.85, color='red', linestyle='--', alpha=0.3, label='Target Sensitivity')
axes[0, 0].axhline(y=0.50, color='blue', linestyle='--', alpha=0.3, label='Target Specificity')
axes[0, 0].set_xlabel('Epoch', fontweight='bold')
axes[0, 0].set_ylabel('Rate', fontweight='bold')
axes[0, 0].set_title('Medical Performance Metrics', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

balanced_acc = (df['Val_Sens'] + df['Val_Spec']) / 2
axes[0, 1].plot(df['Epoch'], balanced_acc, 'g-o', linewidth=2.5)
axes[0, 1].set_xlabel('Epoch', fontweight='bold')
axes[0, 1].set_ylabel('Balanced Accuracy', fontweight='bold')
axes[0, 1].set_title('Balanced Accuracy (Sens+Spec)/2', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df['Epoch'], df['Train_Loss'], 'b-o', label='Train', linewidth=2)
axes[1, 0].plot(df['Epoch'], df['Val_Loss'], 'r-s', label='Validation', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontweight='bold')
axes[1, 0].set_ylabel('Loss', fontweight='bold')
axes[1, 0].set_title('Training vs Validation Loss', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].axis('off')
status_text = f"""
TRAINING SUMMARY
{'='*40}

Total Epochs: {len(df)}
Early Stopping: Triggered at Epoch {len(df)}

FINAL VALIDATION METRICS:
  • Sensitivity: {df['Val_Sens'].iloc[-1]:.2%}
  • Specificity: {df['Val_Spec'].iloc[-1]:.2%}
  • F1 Score: {df['Val_F1'].iloc[-1]:.4f}
  • Accuracy: {df['Val_Acc'].iloc[-1]:.2%}

BEST VALIDATION METRICS:
  • Best Sensitivity: {df['Val_Sens'].max():.2%} (Epoch {df['Val_Sens'].idxmax()+1})
  • Best Specificity: {df['Val_Spec'].max():.2%} (Epoch {df['Val_Spec'].idxmax()+1})
  • Best F1: {df['Val_F1'].max():.4f} (Epoch {df['Val_F1'].idxmax()+1})

LEARNING DYNAMICS:
  • Initial LR: {df['LR'].iloc[0]:.2e}
  • Final LR: {df['LR'].iloc[-1]:.2e}
  • LR Reductions: {len(df['LR'].unique())-1}
"""

axes[1, 1].text(0.1, 0.5, status_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.3))

plt.suptitle('Medical ML Performance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/medical_performance_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: images/medical_performance_analysis.png")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
