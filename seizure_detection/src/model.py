import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) style channel attention.
    Adaptively re-weights channels based on global information.
    """
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (N, C, T) or (N, C, F, T) -> need to handle generalized shape
        # For our use, usually applied on Temporal Feature map (N, C, T)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class TemporalBranch(nn.Module):
    """
    1D CNN to capture temporal dynamics from raw EEG.
    Input: (N, Channels, Time) -> (N, 68, 500)
    """
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 250

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 125

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # -> (N, 64, 1)
            nn.Flatten()
        )
        self.projection = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.net(x)
        return self.projection(x)

class SpectralBranch(nn.Module):
    """
    Hybrid CNN-Transformer for Spectrogram features.
    Input: (N, Channels, Freq, Time) -> (N, 68, 129, 8)
    """
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        # 1. Patch Embedding / Feature Extraction via CNN
        self.cnn_embed = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), # Pool Freq only? (129 -> 64), keep Time (8)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)) # (64 -> 32), keep Time (8)
        )
        
        # Result shape: (N, 64, 32, 8) -> Flatten spatial to sequence
        # Sequence Length = 32 * 8 = 256
        self.hidden_dim = 64
        
        # 2. Transformer Feature Extractor
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        self.projection = nn.Linear(64, out_dim)

    def forward(self, x):
        # x: (N, 68, 129, 8)
        x = self.cnn_embed(x) # -> (N, 64, H', W') e.g. (N, 64, 32, 8)
        
        N, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # -> (N, H*W, C) = (N, 256, 64)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> (N, 257, 64)
        
        # Transformer
        x = self.transformer(x)
        
        # Take CLS output
        x = x[:, 0, :] # -> (N, 64)
        
        return self.projection(x)

class HybridSeizureModel(nn.Module):
    def __init__(self, num_channels=68, num_classes=2):
        super().__init__()
        
        # Branches
        self.temporal = TemporalBranch(in_channels=num_channels)
        self.spectral = SpectralBranch(in_channels=num_channels)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_raw, x_spec):
        t_out = self.temporal(x_raw)
        s_out = self.spectral(x_spec)
        
        combined = torch.cat((t_out, s_out), dim=1)
        return self.fusion(combined)
