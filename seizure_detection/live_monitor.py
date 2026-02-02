import pandas as pd
import time
import os
from datetime import datetime

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_metric(value, width=8):
    """Format metric value to percentage or float"""
    if value > 1:
        return f"{value:>{width}.2f}"
    else:
        return f"{value*100:>{width}.2f}%"

def monitor_training(patient_num=3, refresh_interval=5):
    """
    Monitor training progress in real-time
    
    Args:
        patient_num: Patient number to monitor (default: 3)
        refresh_interval: Seconds between updates (default: 5)
    """
    history_file = f"history_Patient_{patient_num}_improved.csv"
    
    print(f"[*] Monitoring: {history_file}")
    print(f"[*] Refresh interval: {refresh_interval} seconds")
    print(f"[*] Press Ctrl+C to stop\n")
    
    last_epoch = 0
    
    try:
        while True:
            if not os.path.exists(history_file):
                print(f"[!] Waiting for {history_file} to be created...")
                time.sleep(refresh_interval)
                continue
            
            # Read current data
            try:
                df = pd.read_csv(history_file)
                current_epoch = len(df)
                
                if current_epoch > last_epoch or last_epoch == 0:
                    clear_screen()
                    
                    # Header
                    print("=" * 120)
                    print(f"LIVE TRAINING MONITOR - Patient_{patient_num}")
                    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("=" * 120)
                    print(f"\nProgress: {current_epoch}/50 epochs ({current_epoch*2}% complete)")
                    print("-" * 120)
                    
                    # Column headers
                    print(f"{'Epoch':>6} | {'Tr Loss':>9} {'Tr Acc':>9} {'Tr Sens':>9} {'Tr Spec':>9} | "
                          f"{'Val Loss':>9} {'Val Acc':>9} {'Val Sens':>9} {'Val Spec':>9} | {'Val F1':>9} | {'LR':>10}")
                    print("-" * 120)
                    
                    # Show last 10 epochs
                    start_idx = max(0, current_epoch - 10)
                    for idx in range(start_idx, current_epoch):
                        row = df.iloc[idx]
                        epoch = int(row['Epoch'])
                        
                        # Highlight best epochs
                        marker = ""
                        if idx > 0:
                            if row['Val_Acc'] == df['Val_Acc'].max():
                                marker = " [BEST ACC]"
                            elif row['Val_F1'] == df['Val_F1'].max():
                                marker = " [BEST F1]"
                        
                        print(f"{epoch:>6} | "
                              f"{row['Train_Loss']:>9.4f} {format_metric(row['Train_Acc'])} "
                              f"{format_metric(row['Train_Sens'])} {format_metric(row['Train_Spec'])} | "
                              f"{row['Val_Loss']:>9.4f} {format_metric(row['Val_Acc'])} "
                              f"{format_metric(row['Val_Sens'])} {format_metric(row['Val_Spec'])} | "
                              f"{row['Val_F1']:>9.4f} | {row['LR']:>10.2e}{marker}")
                    
                    print("-" * 120)
                    
                    # Summary statistics
                    print("\nCURRENT BEST METRICS:")
                    print(f"  Best Val Accuracy:    {format_metric(df['Val_Acc'].max())} (Epoch {df['Val_Acc'].idxmax() + 1})")
                    print(f"  Best Val Sensitivity: {format_metric(df['Val_Sens'].max())} (Epoch {df['Val_Sens'].idxmax() + 1})")
                    print(f"  Best Val Specificity: {format_metric(df['Val_Spec'].max())} (Epoch {df['Val_Spec'].idxmax() + 1})")
                    print(f"  Best Val F1 Score:    {df['Val_F1'].max():.4f} (Epoch {df['Val_F1'].idxmax() + 1})")
                    
                    # Latest epoch details
                    latest = df.iloc[-1]
                    print(f"\nLATEST EPOCH ({current_epoch}):")
                    print(f"  Validation Accuracy:    {format_metric(latest['Val_Acc'])}")
                    print(f"  Validation Sensitivity: {format_metric(latest['Val_Sens'])} (seizure detection)")
                    print(f"  Validation Specificity: {format_metric(latest['Val_Spec'])} (non-seizure detection)")
                    print(f"  Validation F1 Score:    {latest['Val_F1']:.4f}")
                    print(f"  Overfitting Gap:        {format_metric(latest['Acc_Gap'])}")
                    print(f"  Learning Rate:          {latest['LR']:.2e}")
                    
                    # Progress bar
                    progress = current_epoch / 50
                    bar_length = 50
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\n[{bar}] {current_epoch}/50 epochs")
                    
                    print("\n" + "=" * 120)
                    print(f"Refreshing every {refresh_interval} seconds... (Ctrl+C to stop)")
                    
                    last_epoch = current_epoch
                
                time.sleep(refresh_interval)
                
            except Exception as e:
                print(f"[!] Error reading file: {e}")
                time.sleep(refresh_interval)
                
    except KeyboardInterrupt:
        print("\n\n[*] Monitoring stopped by user")
        print(f"[*] Final epoch count: {last_epoch}/50")

if __name__ == "__main__":
    import sys
    
    # Get patient number from command line or default to 3
    patient_num = 3
    if len(sys.argv) > 1:
        try:
            patient_num = int(sys.argv[1])
        except:
            print(f"Usage: python live_monitor.py [patient_number]")
            print(f"Using default: Patient_3")
    
    monitor_training(patient_num=patient_num, refresh_interval=5)
