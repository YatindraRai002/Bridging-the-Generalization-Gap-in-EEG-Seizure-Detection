import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create images directory if not exists
os.makedirs('images', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 9

# Load all patient histories
patients = []
import glob
# Find all improved history files and sort them numerically by patient ID
history_files = sorted(glob.glob('history_Patient_*_improved.csv'), 
                      key=lambda x: int(x.split('_')[2]) if x.split('_')[2].isdigit() else 0)

for file in history_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        # Extract patient ID/name from filename
        patient_name = file.replace('history_', '').replace('_improved.csv', '')
        df['Patient'] = patient_name
        patients.append(df)

if not patients:
    print("[!] No training history files found!")
    exit(1)

print(f"[*] Loaded {len(patients)} patient histories")

# Create comprehensive figure
fig = plt.figure(figsize=(18, 14))

# 1. Validation Accuracy Comparison
ax1 = plt.subplot(3, 3, 1)
for i, df in enumerate(patients):
    plt.plot(df['Epoch'], df['Val_Acc'], marker='o', label=df['Patient'].iloc[0], linewidth=2)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Validation Accuracy', fontweight='bold')
plt.title('Validation Accuracy Across Patients', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Validation Sensitivity (Critical for Medical)
ax2 = plt.subplot(3, 3, 2)
for i, df in enumerate(patients):
    plt.plot(df['Epoch'], df['Val_Sens'], marker='s', label=df['Patient'].iloc[0], linewidth=2)
plt.axhline(y=0.85, color='red', linestyle='--', alpha=0.3, label='Target (85%)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Validation Sensitivity', fontweight='bold')
plt.title('Sensitivity - Seizure Detection Rate', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Validation Specificity
ax3 = plt.subplot(3, 3, 3)
for i, df in enumerate(patients):
    plt.plot(df['Epoch'], df['Val_Spec'], marker='^', label=df['Patient'].iloc[0], linewidth=2)
plt.axhline(y=0.50, color='green', linestyle='--', alpha=0.3, label='Target (50%)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Validation Specificity', fontweight='bold')
plt.title('Specificity - Non-Seizure Detection Rate', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 4. F1 Score Progression
ax4 = plt.subplot(3, 3, 4)
for i, df in enumerate(patients):
    plt.plot(df['Epoch'], df['Val_F1'], marker='d', label=df['Patient'].iloc[0], linewidth=2)
plt.axhline(y=0.55, color='orange', linestyle='--', alpha=0.3, label='Target (0.55)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('F1 Score', fontweight='bold')
plt.title('F1 Score Evolution', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Training vs Validation Loss
ax5 = plt.subplot(3, 3, 5)
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
for i, df in enumerate(patients):
    patient_name = df['Patient'].iloc[0]
    plt.plot(df['Epoch'], df['Train_Loss'], '--', color=colors[i], alpha=0.5, label=f'{patient_name} Train')
    plt.plot(df['Epoch'], df['Val_Loss'], '-', color=colors[i], linewidth=2, label=f'{patient_name} Val')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.title('Training vs Validation Loss', fontweight='bold', fontsize=11)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 6. Overfitting Gap
ax6 = plt.subplot(3, 3, 6)
for i, df in enumerate(patients):
    plt.plot(df['Epoch'], df['Acc_Gap'], marker='o', label=df['Patient'].iloc[0], linewidth=2)
plt.axhline(y=0.15, color='red', linestyle='--', alpha=0.3, label='Warning (15%)')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Overfitting Gap', fontweight='bold')
plt.title('Overfitting Gap (Train - Val Acc)', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Learning Rate Schedule
ax7 = plt.subplot(3, 3, 7)
for i, df in enumerate(patients):
    plt.plot(df['Epoch'], df['LR'], marker='o', label=df['Patient'].iloc[0], linewidth=2)
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Learning Rate', fontweight='bold')
plt.title('Learning Rate Schedule', fontweight='bold', fontsize=11)
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Sensitivity-Specificity Scatter
ax8 = plt.subplot(3, 3, 8)
colors_scatter = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
for i, df in enumerate(patients):
    # Plot final epoch as larger point
    plt.scatter(df['Val_Spec'], df['Val_Sens'], c=colors_scatter[i], 
               alpha=0.5, s=30, label=df['Patient'].iloc[0])
    # Highlight final epoch
    plt.scatter(df['Val_Spec'].iloc[-1], df['Val_Sens'].iloc[-1], 
               c=colors_scatter[i], s=200, marker='*', edgecolors='black', linewidths=2)
plt.xlabel('Specificity', fontweight='bold')
plt.ylabel('Sensitivity', fontweight='bold')
plt.title('Sensitivity-Specificity Trade-off\n(★ = Final Epoch)', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Summary Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_data = [['Patient', 'Epochs', 'Best Acc', 'Best Sens', 'Best Spec', 'Best F1']]
for df in patients:
    patient_name = df['Patient'].iloc[0]
    summary_data.append([
        patient_name,
        f"{len(df)}",
        f"{df['Val_Acc'].max():.3f}",
        f"{df['Val_Sens'].max():.3f}",
        f"{df['Val_Spec'].max():.3f}",
        f"{df['Val_F1'].max():.3f}"
    ])

table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate rows
for i in range(1, len(summary_data)):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

plt.suptitle('Multi-Patient LOSO Training Analysis (Cross-Subject Evaluation)', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('images/multi_patient_training_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: images/multi_patient_training_analysis.png")

# Create individual patient detailed plots
for patient_idx, df in enumerate(patients):
    patient_name = df['Patient'].iloc[0]
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Medical metrics
    axes[0, 0].plot(df['Epoch'], df['Val_Sens'], 'r-o', label='Sensitivity', linewidth=2)
    axes[0, 0].plot(df['Epoch'], df['Val_Spec'], 'b-s', label='Specificity', linewidth=2)
    axes[0, 0].axhline(y=0.85, color='red', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(y=0.50, color='blue', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('Rate', fontweight='bold')
    axes[0, 0].set_title(f'{patient_name} - Medical Metrics', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy and F1
    ax_acc = axes[0, 1]
    ax_f1 = ax_acc.twinx()
    ax_acc.plot(df['Epoch'], df['Val_Acc'], 'g-o', label='Accuracy', linewidth=2)
    ax_f1.plot(df['Epoch'], df['Val_F1'], 'orange', linestyle='--', marker='s', label='F1 Score', linewidth=2)
    ax_acc.set_xlabel('Epoch', fontweight='bold')
    ax_acc.set_ylabel('Accuracy', fontweight='bold', color='g')
    ax_f1.set_ylabel('F1 Score', fontweight='bold', color='orange')
    ax_acc.set_title(f'{patient_name} - Accuracy & F1', fontweight='bold')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc='upper left')
    ax_f1.legend(loc='upper right')
    
    # Loss curves
    axes[1, 0].plot(df['Epoch'], df['Train_Loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[1, 0].plot(df['Epoch'], df['Val_Loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontweight='bold')
    axes[1, 0].set_ylabel('Loss', fontweight='bold')
    axes[1, 0].set_title(f'{patient_name} - Loss Curves', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary stats
    axes[1, 1].axis('off')
    stats_text = f"""
{patient_name} TRAINING SUMMARY
{'='*35}

Total Epochs: {len(df)}
Final LR: {df['LR'].iloc[-1]:.2e}

FINAL METRICS:
  Accuracy:    {df['Val_Acc'].iloc[-1]:.2%}
  Sensitivity: {df['Val_Sens'].iloc[-1]:.2%}
  Specificity: {df['Val_Spec'].iloc[-1]:.2%}
  F1 Score:    {df['Val_F1'].iloc[-1]:.4f}

BEST METRICS:
  Best Acc:    {df['Val_Acc'].max():.2%} (Epoch {df['Val_Acc'].idxmax()+1})
  Best Sens:   {df['Val_Sens'].max():.2%} (Epoch {df['Val_Sens'].idxmax()+1})
  Best Spec:   {df['Val_Spec'].max():.2%} (Epoch {df['Val_Spec'].idxmax()+1})
  Best F1:     {df['Val_F1'].max():.4f} (Epoch {df['Val_F1'].idxmax()+1})

OVERFITTING:
  Final Gap:   {df['Acc_Gap'].iloc[-1]:.2%}
  Avg Gap:     {df['Acc_Gap'].mean():.2%}
"""
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'{patient_name} Detailed Training Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'images/{patient_name}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: images/{patient_name}_detailed_analysis.png")

print("\n" + "="*60)
print("ALL VISUALIZATIONS COMPLETE")
print("="*60)
print(f"\nGenerated {len(patients) + 1} visualization files:")
print("  - multi_patient_training_analysis.png (comparison)")
for df in patients:
    print(f"  - {df['Patient'].iloc[0]}_detailed_analysis.png")
