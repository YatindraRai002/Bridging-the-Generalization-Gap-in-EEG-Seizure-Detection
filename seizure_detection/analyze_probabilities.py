
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns

from src.dataset import EEGDataset, get_all_file_paths, MAX_CHANNELS
from src.models.baseline_cnn import BaselineCNN

# Configuration
DATA_ROOT = r"c:\Users\Asus\Downloads\clips\Volumes\Seagate\seizure_detection\competition_data\clips"
MODEL_PATH = "model_Patient_1_balanced.pth"  # Current training model
TEST_SUBJECT = "Patient_1"
BATCH_SIZE = 64

def analyze_probability_distribution(model_path, show_plot=True):
    """Analyze probability distributions to check if model can discriminate"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"🔬 PROBABILITY DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    
    # Load dataset
    files, labels, subjects = get_all_file_paths(DATA_ROOT)
    files = np.array(files)
    labels = np.array(labels)
    subjects = np.array(subjects)
    
    # Get test subject data
    test_mask = (subjects == TEST_SUBJECT)
    X_test, y_test = files[test_mask], labels[test_mask]
    
    print(f"\nTest samples: {len(X_test)}")
    print(f"  Seizure: {sum(y_test == 1)}")
    print(f"  Non-Seizure: {sum(y_test == 0)}")
    
    # Load model
    model = BaselineCNN(num_channels=MAX_CHANNELS).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"\n Model file not found: {model_path}")
        print(f"   Training may still be in progress...")
        return None
    
    model.eval()
    
    # Get predictions
    test_ds = EEGDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels_batch in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            seizure_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(seizure_probs)
            all_labels.extend(labels_batch.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Separate by class
    seizure_probs = all_probs[all_labels == 1]
    non_seizure_probs = all_probs[all_labels == 0]
    
    # Statistics
    print(f"\n{'='*80}")
    print(f"📊 PROBABILITY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\n🔴 Seizure Samples (n={len(seizure_probs)}):")
    print(f"   Mean:   {seizure_probs.mean():.4f}")
    print(f"   Median: {np.median(seizure_probs):.4f}")
    print(f"   Min:    {seizure_probs.min():.4f}")
    print(f"   Max:    {seizure_probs.max():.4f}")
    print(f"   Std:    {seizure_probs.std():.4f}")
    
    print(f"\n🟢 Non-Seizure Samples (n={len(non_seizure_probs)}):")
    print(f"   Mean:   {non_seizure_probs.mean():.4f}")
    print(f"   Median: {np.median(non_seizure_probs):.4f}")
    print(f"   Min:    {non_seizure_probs.min():.4f}")
    print(f"   Max:    {non_seizure_probs.max():.4f}")
    print(f"   Std:    {non_seizure_probs.std():.4f}")
    
    # Separation analysis
    overlap = np.sum((seizure_probs.min() <= non_seizure_probs) & (non_seizure_probs <= seizure_probs.max()))
    overlap_pct = (overlap / len(non_seizure_probs)) * 100
    
    gap = seizure_probs.min() - non_seizure_probs.max()
    
    print(f"\n{'='*80}")
    print(f"🎯 SEPARATION ANALYSIS")
    print(f"{'='*80}")
    print(f"   Gap (Seizure_min - NonSeizure_max): {gap:.4f}")
    print(f"   Overlap: {overlap}/{len(non_seizure_probs)} ({overlap_pct:.1f}%)")
    
    # Diagnosis
    print(f"\n{'='*80}")
    print(f"🏥 DIAGNOSIS")
    print(f"{'='*80}")
    
    if gap > 0.1:
        print(f"   ✅ EXCELLENT SEPARATION")
        print(f"      Distributions are well separated")
        print(f"      Threshold tuning will work!")
        status = "excellent"
    elif gap > 0:
        print(f"   🟢 GOOD SEPARATION")
        print(f"      Distributions barely touch")
        print(f"      Threshold tuning should work")
        status = "good"
    elif non_seizure_probs.mean() < seizure_probs.mean() - 0.1:
        print(f"   🟡 PARTIAL SEPARATION")
        print(f"      Distributions overlap but means differ")
        print(f"      Threshold tuning may help")
        status = "partial"
    elif non_seizure_probs.mean() < seizure_probs.mean():
        print(f"   🟠 WEAK SEPARATION")
        print(f"      Distributions heavily overlap")
        print(f"      Threshold tuning limited effectiveness")
        status = "weak"
    else:
        print(f"   🔴 NO SEPARATION (SATURATED)")
        print(f"      Model predicts high probability for everything")
        print(f"      Threshold tuning WILL NOT WORK")
        print(f"      ⚠️ MUST reduce seizure weight and retrain")
        status = "saturated"
    
    # Recommendations
    print(f"\n{'='*80}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if status in ["excellent", "good"]:
        print(f"   ✅ Model is ready for threshold optimization")
        print(f"   ✅ Try thresholds between {non_seizure_probs.max():.2f} and {seizure_probs.min():.2f}")
    elif status == "partial":
        print(f"   🟡 Model shows promise but needs improvement")
        print(f"   → Try reducing seizure weight slightly (current - 2x)")
        print(f"   → Or continue training for more epochs")
    elif status == "weak":
        print(f"   ⚠️ Model needs significant improvement")
        print(f"   → Reduce seizure weight by 25-30%")
        print(f"   → Check if model is overfitting")
    else:  # saturated
        print(f"   🔴 Model is saturated - MUST retrain")
        print(f"   → Reduce seizure weight to 12-15x (currently likely 16-24x)")
        print(f"   → Check training curves for overfitting")
        print(f"   → Consider adding more regularization")
    
    # Create visualization
    if show_plot:
        create_probability_plots(seizure_probs, non_seizure_probs, status)
    
    return {
        'seizure_probs': seizure_probs,
        'non_seizure_probs': non_seizure_probs,
        'gap': gap,
        'overlap_pct': overlap_pct,
        'status': status
    }

def create_probability_plots(seizure_probs, non_seizure_probs, status):
    """Create probability distribution visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Probability Distribution Analysis - Status: {status.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Overlapping histograms
    ax1 = axes[0, 0]
    ax1.hist(non_seizure_probs, bins=50, alpha=0.6, label='Non-Seizure', color='green', edgecolor='black')
    ax1.hist(seizure_probs, bins=50, alpha=0.6, label='Seizure', color='red', edgecolor='black')
    ax1.axvline(non_seizure_probs.mean(), color='green', linestyle='--', linewidth=2, label=f'Non-Sz Mean: {non_seizure_probs.mean():.3f}')
    ax1.axvline(seizure_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'Sz Mean: {seizure_probs.mean():.3f}')
    ax1.set_xlabel('Seizure Probability', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Probability Distributions (Overlapping)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Side-by-side box plots
    ax2 = axes[0, 1]
    data_to_plot = [non_seizure_probs, seizure_probs]
    bp = ax2.boxplot(data_to_plot, labels=['Non-Seizure', 'Seizure'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax2.set_ylabel('Seizure Probability', fontsize=12)
    ax2.set_title('Probability Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Density plot
    ax3 = axes[1, 0]
    from scipy.stats import gaussian_kde
    
    if len(np.unique(non_seizure_probs)) > 1:
        kde_non_sz = gaussian_kde(non_seizure_probs)
        x_range = np.linspace(0, 1, 1000)
        ax3.plot(x_range, kde_non_sz(x_range), label='Non-Seizure', color='green', linewidth=2)
    
    if len(np.unique(seizure_probs)) > 1:
        kde_sz = gaussian_kde(seizure_probs)
        ax3.plot(x_range, kde_sz(x_range), label='Seizure', color='red', linewidth=2)
    
    ax3.fill_between(x_range, 0, kde_non_sz(x_range) if len(np.unique(non_seizure_probs)) > 1 else 0, 
                      alpha=0.3, color='green')
    ax3.fill_between(x_range, 0, kde_sz(x_range) if len(np.unique(seizure_probs)) > 1 else 0, 
                      alpha=0.3, color='red')
    ax3.set_xlabel('Seizure Probability', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Probability Density Functions', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_non_sz = np.sort(non_seizure_probs)
    sorted_sz = np.sort(seizure_probs)
    ax4.plot(sorted_non_sz, np.arange(len(sorted_non_sz)) / len(sorted_non_sz), 
             label='Non-Seizure', color='green', linewidth=2)
    ax4.plot(sorted_sz, np.arange(len(sorted_sz)) / len(sorted_sz), 
             label='Seizure', color='red', linewidth=2)
    ax4.set_xlabel('Seizure Probability', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Distribution Functions', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plots saved to probability_distribution_analysis.png")
    plt.close()

if __name__ == "__main__":
    analyze_probability_distribution(MODEL_PATH)
