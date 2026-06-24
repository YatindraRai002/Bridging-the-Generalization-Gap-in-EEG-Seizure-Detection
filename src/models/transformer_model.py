import torch
import torch.nn as nn

class EEGTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for EEG.
    1. Channel-wise CNN for temporal feature extraction.
    2. Transformer Encoder across Channels (Spatial Attention).
    """
    def __init__(self, num_channels, num_classes=2, d_model=64, nhead=4, num_layers=2):
        super(EEGTransformer, self).__init__()
        
        # 1. Temporal Feature Extraction (Per Channel)
        # We share weights across channels (Grouped Conv or reshape)
        # Input: (B, C, T) -> reshape to (B*C, 1, T) for shared weights
        
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # T -> T/2
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # T/2 -> T/4
            
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # (B*C, d_model, 1)
        )
        
        self.d_model = d_model
        
        # 2. Transformer Encoder (Spatial)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classification
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Reshape for Temporal CNN (Shared weights)
        x = x.view(B * C, 1, T)
        
        # Extract Temporal Features
        x = self.temporal_cnn(x) # (B*C, d_model, 1)
        x = x.view(B, C, self.d_model) # (B, C, d_model)
        
        # Transformer (Spatial Attention across Channels)
        # Treat Channels as Tokens
        # We can add learnable positional encoding for channels if desired, 
        # but since channels vary, maybe no PE is better (permutation invariant).
        
        x = self.transformer_encoder(x) # (B, C, d_model)
        
        # Global Pooling (Average across channels)
        x = x.mean(dim=1) # (B, d_model)
        
        out = self.fc(x)
        return out
