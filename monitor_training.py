
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import FuncAnimation

def monitor_training(history_file, update_interval=5):

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Real-Time Training Monitor', fontsize=16, fontweight='bold')

    def update_plots(frame):
        if not os.path.exists(history_file):
            return

        try:
            df = pd.read_csv(history_file)

            if len(df) == 0:
                return

            for ax in axes.flat:
                ax.clear()

            epochs = df['Epoch']

            ax1 = axes[0, 0]
            ax1.plot(epochs, df['Train_Loss'], 'o-', label='Train Loss', linewidth=2, markersize=6, color='#3498db')
            ax1.plot(epochs, df['Val_Loss'], 's-', label='Val Loss', linewidth=2, markersize=6, color='#e74c3c')
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            ax2.plot(epochs, df['Train_Acc']*100, 'o-', label='Train Acc', linewidth=2, markersize=6, color='#2ecc71')
            ax2.plot(epochs, df['Val_Acc']*100, 's-', label='Val Acc', linewidth=2, markersize=6, color='#f39c12')
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Accuracy (%)', fontsize=11)
            ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            if 'Acc_Gap' in df.columns:
                ax3 = axes[0, 2]
                ax3.plot(epochs, df['Acc_Gap']*100, 'o-', linewidth=2, markersize=6, color='#9b59b6')
                ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
                ax3.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='15% threshold')
                ax3.set_xlabel('Epoch', fontsize=11)
                ax3.set_ylabel('Gap (%)', fontsize=11)
                ax3.set_title('Overfitting Gap (Train - Val Acc)', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 0]
            ax4.plot(epochs, df['Val_Sens']*100, 'o-', label='Sensitivity', linewidth=2, markersize=6, color='#2ecc71')
            ax4.plot(epochs, df['Val_Spec']*100, 's-', label='Specificity', linewidth=2, markersize=6, color='#e74c3c')
            ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
            ax4.set_xlabel('Epoch', fontsize=11)
            ax4.set_ylabel('Percentage (%)', fontsize=11)
            ax4.set_title('Sensitivity vs Specificity', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            ax5 = axes[1, 1]
            ax5.plot(epochs, df['Val_F1'], 'o-', linewidth=2, markersize=6, color='#9b59b6')
            ax5.set_xlabel('Epoch', fontsize=11)
            ax5.set_ylabel('F1 Score', fontsize=11)
            ax5.set_title('F1 Score Progression', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)

            if 'LR' in df.columns:
                ax6 = axes[1, 2]
                ax6.plot(epochs, df['LR'], 'o-', linewidth=2, markersize=6, color='#e67e22')
                ax6.set_xlabel('Epoch', fontsize=11)
                ax6.set_ylabel('Learning Rate', fontsize=11)
                ax6.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
                ax6.set_yscale('log')
                ax6.grid(True, alpha=0.3)

            latest = df.iloc[-1]
            stats_text = f"Epoch {int(latest['Epoch'])} | Val Acc: {latest['Val_Acc']*100:.2f}% | Val F1: {latest['Val_F1']:.4f}"
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        except Exception as e:
            print(f"Error updating plots: {e}")

    ani = FuncAnimation(fig, update_plots, interval=update_interval*1000, cache_frame_data=False)

    plt.show()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        history_file = sys.argv[1]
    else:
        history_file = "history_Patient_1_improved.csv"

    print(f"📊 Monitoring training progress from: {history_file}")
    print(f"🔄 Updating every 5 seconds...")
    print(f"❌ Close the plot window to stop monitoring")

    monitor_training(history_file)
