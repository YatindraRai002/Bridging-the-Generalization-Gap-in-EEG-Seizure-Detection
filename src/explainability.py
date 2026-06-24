import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import EEGDataset

def visualize_channel_attention(model, data, channels_labels=None, device='cpu'):
    """
    Visualize which channels the model is focusing on.
    Since we used a Transformer Encoder, we can look at the self-attention weights.
    However, standard nn.TransformerEncoder doesn't easily expose weights unless we use a hook
    or custom implementation.

    Alternatively, for experimental saliency:
    We can use Gradient-based Saliency (Input Gradients).
    """
    model.eval()
    data = data.to(device)
    data.requires_grad_()

    output = model(data)

    target_score = output[0, 1]

    model.zero_grad()
    target_score.backward()

    gradients = data.grad.data.cpu().numpy()[0]

    channel_importance = np.mean(np.abs(gradients), axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.arange(len(channel_importance)), y=channel_importance)
    plt.title("Constraint-Based Saliency: Channel Importance")
    plt.xlabel("Channel Index")
    plt.ylabel("Gradient Magnitude (Importance)")
    plt.savefig("saliency_map.png")
    print("Saved saliency_map.png")

    return channel_importance
