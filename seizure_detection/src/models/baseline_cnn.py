import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Standard 1D CNN Baseline for EEG Seizure Detection.
    A simple, robust architecture to serve as a benchmark.
    
    Architecture:
    - 3 Convolutional Blocks (Conv1D + BN + ReLU + MaxPool)
    - Global Average Pooling
    - Fully Connected Classification Head
    """
    def __init__(self, num_channels, num_classes=2):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(num_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 2
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x: (B, C, T)
        x = self.features(x)
        x = self.classifier(x)
        return x
