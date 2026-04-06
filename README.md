# 🧠 EEG-ViT: Bridging the Generalization Gap in Seizure Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CHB--MIT-2E8B57?style=for-the-badge)
![Protocol](https://img.shields.io/badge/Eval-LOSO%2022--Fold-FF6B35?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A Hybrid Vision Transformer with Adversarial Subject Disentanglement for Cross-Patient EEG Seizure Detection**


</div>

---

## 📌 The Problem in One Picture

```
Within-Subject Evaluation (most papers)        LOSO Evaluation (clinical reality)
─────────────────────────────────────          ─────────────────────────────────
  Train: Patient 1, 2, 3 ... 22                 Train: Patient 1, 2, 3 ... 21
  Test:  Patient 1, 2, 3 ... 22   ✅            Test:  Patient 22 (NEVER seen) ✅

  Accuracy looks great on paper                 Baseline CNN: 24.82% sensitivity
  BUT model memorises patient identity          1001 false alarms, 70 true detections
  → Silently fails on new patients              → Clinically UNSAFE
```

> **Core insight:** Conventional deep learning EEG models encode *who the patient is* more than *whether a seizure is happening*. This repo fixes that.

---

## 🔬 What This Project Does

This repository implements **EEG-ViT**, a hybrid architecture that:

1. **Diagnoses** why cross-patient generalization fails — using Silhouette Score analysis of learned embedding spaces
2. **Quantifies** the failure formally via the per-subject Generalization Gap metric $G_s$
3. **Corrects** the failure through adversarial subject disentanglement, pushing the encoder to learn seizure-discriminative rather than patient-correlated representations

---

## 📊 Key Results (Patient 7, CHB-MIT, LOSO)

| Metric | Baseline CNN | EEG-ViT (Ours) | Improvement |
|--------|-------------|----------------|-------------|
| Accuracy | 65.55% | **87.67%** | +22.12% |
| Sensitivity | 24.82% | **84.04%** | +59.22% |
| Specificity | 69.10% | **87.99%** | +18.89% |
| F1 Score | 0.1035 | **0.5220** | +0.4185 |
| False Positives | 1001 | **389** | −61.1% |
| False Negatives | 212 | **45** | −78.8% |
| False alarms per true seizure | 14.3 | **1.64** | **8.7× reduction** |

### Representation Space (Why It Works)

| Metric | Baseline CNN | EEG-ViT |
|--------|-------------|---------|
| $S_{\text{patient}}$ (Silhouette) | 0.74 | 0.57 |
| $S_{\text{seizure}}$ (Silhouette) | 0.21 | 0.39 |
| Ratio $S_p / S_s$ | **3.52** (identity-dominated) | **1.48** (seizure-organised) |

The baseline embedding is 3.52× more organised around *who the patient is* than *whether they are seizing*. EEG-ViT compresses this to 1.48.

---

## 🏗️ Architecture Overview

```
Raw EEG (23 ch × 1280 samples)
        │
        ▼
┌───────────────────┐
│  Bandpass Filter  │  0.5 – 40 Hz  (removes DC drift + EMG)
│  Z-score Norm     │  Per-channel, training stats only
│  5-sec Windows    │  Non-overlapping, ictal/interictal labels
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────┐
│         CNN Encoder  f_θ           │
│  Conv1: Depthwise Spatial (32 f)   │  ← Electrode-level spatial features
│  Conv2: Temporal BN+ReLU  (64 f)   │  ← Short-range temporal patterns
│  Conv3: Temporal BN+ReLU  (128 f)  │  ← Higher-level temporal structure
└────────────────────────────────────┘
         │   {h_t} frame-level features
         ▼
┌────────────────────────────────────┐
│    Temporal Attention Gate  α_t    │  ← Soft-weights ictal vs quiet frames
│    Latent vector  z  (d-dim)       │
└──────────────┬─────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐   ┌──────────────────┐
│ Seizure     │   │ Patient Identity │
│ Head  g_s   │   │ Head  g_p + GRL  │  ← Gradient Reversal Layer
│ (softmax)   │   │ (adversarial)    │
└──────┬──────┘   └────────┬─────────┘
       │                   │
       ▼                   ▼
  L_focal              L_patient
       │                   │
       └────────┬──────────┘
                ▼
     L_total = L_focal − λ · L_patient
                │
                ▼
             AdamW update
```

---

## 🧮 Mathematical Formulation

### Focal Loss (Class Imbalance)
Seizure windows make up <5% of all segments. Standard cross-entropy still lets easy interictal examples dominate gradients even with class weights. Focal loss suppresses them:

$$\mathcal{L}_{\text{focal}} = -\sum_{i=1}^{N} w_{y_i}(1 - \hat{p}_{y_i})^{\gamma} \log \hat{p}_{y_i}, \quad \gamma = 2$$

High-confidence predictions contribute near-zero gradient. Training concentrates on the hard perictal boundary windows where errors originate.

### Adversarial Subject Disentanglement
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} - \lambda \cdot \mathcal{L}_{\text{patient}}$$

$$\mathcal{L}_{\text{patient}} = -\sum_{i} \log p(s_i \mid z_i;\, g_p)$$

- $g_p$ is trained to **maximise** patient identity classification from $z$
- $f_\theta$ is updated (via negated gradients at GRL) to **minimise** patient classification
- At Nash equilibrium: $z$ retains only features that are ictal-predictive **and** patient-invariant

### Temporal Attention Gate
$$\alpha_t = \frac{\exp(v^\top \tanh(W_a h_t + b_a))}{\sum_{t'} \exp(v^\top \tanh(W_a h_{t'} + b_a))}, \qquad z = \sum_t \alpha_t h_t$$

Assigns higher weight to time steps exhibiting ictal oscillatory patterns within each 5-second window.

### Generalization Gap Metric
$$G_s = \text{Acc}_{\text{within}}(s) - \text{Acc}_{\text{LOSO}}(s)$$

$$\bar{G} = \frac{1}{S}\sum_s G_s, \qquad \sigma^2_G = \frac{1}{S}\sum_s (G_s - \bar{G})^2$$

Baseline results: $\bar{G} = 0.23$, $\sigma_G = 0.14$ — high variance means failure modes are qualitatively different per patient, making no single threshold safe.

---

## 📁 Repository Structure

```
eeg-vit-seizure/
│
├── data/
│   ├── download_chbmit.py          # PhysioNet CHB-MIT download script
│   ├── preprocess.py               # Bandpass filter + z-score + windowing
│   └── loso_splits.py              # Generate 22-fold LOSO partitions
│
├── models/
│   ├── baseline_cnn.py             # 3-block CNN baseline
│   ├── eeg_vit.py                  # Hybrid EEG-ViT (main model)
│   ├── attention_gate.py           # Temporal attention gate module
│   ├── grl.py                      # Gradient Reversal Layer
│   └── heads.py                    # Seizure head g_s + patient head g_p
│
├── training/
│   ├── focal_loss.py               # Focal loss with inverse-frequency weights
│   ├── trainer.py                  # Training loop with dual objectives
│   └── early_stopping.py           # Patience-based early stopping
│
├── evaluation/
│   ├── loso_eval.py                # 22-fold LOSO evaluation engine
│   ├── metrics.py                  # Accuracy, F1, sensitivity, specificity
│   ├── generalization_gap.py       # G_s and σ²_G computation
│   └── silhouette_analysis.py      # Silhouette Score on latent embeddings
│
├── visualisation/
│   ├── confusion_matrix.py         # Side-by-side confusion matrix plots
│   ├── tsne_embedding.py           # t-SNE of learned representations
│   └── bar_charts.py               # Cross-model comparison charts
│
├── configs/
│   └── default.yaml                # All hyperparameters
│
├── train.py                        # Entry point: train EEG-ViT
├── evaluate.py                     # Entry point: LOSO evaluation
├── requirements.txt
└── README.md
```

---

## ⚙️ Model Configuration

```yaml
# configs/default.yaml

model:
  conv1_filters: 32       # Depthwise spatial, kernel 1×3
  conv2_filters: 64       # Temporal, BN + ReLU
  conv3_filters: 128      # Temporal, BN + ReLU
  patch_size: 16
  embed_dim: 128
  vit_layers: 4
  vit_heads: 4
  vit_mlp_dim: 256
  attn_gate_da: 64
  attn_gate_dh: 128
  dropout: 0.5
  fc_units: 256

training:
  learning_rate: 5.0e-5
  batch_size: 64
  max_epochs: 50
  optimizer: adamw
  weight_decay: 1.0e-4
  early_stopping_patience: 15
  focal_gamma: 2
  disentanglement_lambda: 0.1

data:
  sampling_rate: 256       # Hz
  n_channels: 23
  window_seconds: 5
  window_samples: 1280     # 5 × 256
  bandpass_low: 0.5        # Hz
  bandpass_high: 40.0      # Hz
  n_subjects: 22

evaluation:
  protocol: loso
  n_folds: 22
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-org/eeg-vit-seizure.git
cd eeg-vit-seizure
pip install -r requirements.txt
```

### 2. Download CHB-MIT Dataset

```bash
# Requires PhysioNet account (free registration)
python data/download_chbmit.py --output_dir ./data/chbmit
```

Or download manually from [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) and place under `./data/chbmit/`.

### 3. Preprocess

```bash
python data/preprocess.py \
    --data_dir ./data/chbmit \
    --output_dir ./data/processed \
    --window_sec 5 \
    --bandpass 0.5 40
```

### 4. Train

```bash
# Train EEG-ViT with focal loss + adversarial disentanglement
python train.py --config configs/default.yaml --model eeg_vit

# Train baseline CNN for comparison
python train.py --config configs/default.yaml --model baseline_cnn
```

### 5. Evaluate (LOSO)

```bash
# Full 22-fold LOSO evaluation
python evaluate.py \
    --checkpoint checkpoints/eeg_vit_best.pt \
    --protocol loso \
    --output_dir results/
```

### 6. Analyse Representations

```bash
# Silhouette Score analysis
python evaluation/silhouette_analysis.py --checkpoint checkpoints/eeg_vit_best.pt

# t-SNE embedding visualisation
python visualisation/tsne_embedding.py --checkpoint checkpoints/eeg_vit_best.pt --subject 7

# Generalisation gap across all patients
python evaluation/generalization_gap.py --results_dir results/
```

---

## 📦 Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
mne>=1.4.0           # EEG I/O and filtering
pyyaml>=6.0
tqdm>=4.65.0
wfdb>=4.1.0          # PhysioNet record reading
```

---

## 🗃️ Dataset

**CHB-MIT Scalp EEG Database** — Physionet  
Shoeb, A.H. (2009). *Application of machine learning to epileptic seizure onset detection and treatment*. Ph.D. thesis, MIT.

| Parameter | Value |
|-----------|-------|
| Subjects | 22 paediatric patients |
| Sampling rate | 256 Hz |
| Channels | 23 electrodes |
| Total seizures | 198 annotated events |
| Window length | 5 seconds |
| Class ratio | <5% ictal (severe imbalance) |

LOSO cross-validation: each of the 22 subjects is held out in turn. No patient contributes to both training and test in any fold.

---

## 📉 Per-Patient Generalization Gap

Baseline CNN results across subjects demonstrate why aggregate accuracy is misleading:

| Patient | Accuracy | Sensitivity | $G_s$ | Failure Mode |
|---------|----------|-------------|--------|--------------|
| P1 | 52.9% | 100.0% | 0.36 | Specificity collapse (predicts ictal for everything) |
| P2 | 89.1% | 53.0% | 0.02 | Near-perfect transfer |
| P3 | 65.2% | 33.0% | 0.25 | Moderate boundary failure |
| P7 | 65.6% | 24.8% | 0.33 | Decision boundary failure |
| **Mean** | **66.7%** | **63.0%** | **0.23** | |
| **Std** | **15.2%** | **27.0%** | **0.14** | |

$\sigma_G = 0.14$ means failure modes are qualitatively different per patient. No fixed threshold protects all patients simultaneously.

---

## 🔍 How the Two Objectives Work Together

```
FOCAL LOSS                          GRADIENT REVERSAL
─────────────────────────────────   ──────────────────────────────────
Operates at: optimisation level     Operates at: representation level

Problem it solves:                  Problem it solves:
  Easy interictal samples dominate    Even with balanced gradients,
  gradient signal, suppressing        encoder encodes patient identity
  learning on hard perictal cases     because it correlates with ictal
                                      state within each training patient

Mechanism:                          Mechanism:
  (1 - p̂)^γ modulating factor        g_p maximises patient classification
  → near-zero gradient for           f_θ minimises it (negated gradients)
    high-confidence predictions       → minimax forces patient-invariant
                                        ictal features at equilibrium

Effect:                             Effect:
  Corrects WHAT the model trains on   Corrects WHAT the encoder encodes
```

These objectives are **orthogonal**, not redundant. Both are needed.

---

## 📈 Silhouette Score Interpretation

The Silhouette Score $s(i) \in [-1, 1]$ measures cluster cohesion vs separation:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean distance to nearest other cluster.

| Score computed over | Baseline CNN | EEG-ViT | Ideal |
|--------------------|-------------|---------|-------|
| Patient identity clusters | **0.74** (tight) | 0.57 | → 0.0 |
| Seizure state clusters | 0.21 (loose) | **0.39** | → 1.0 |
| Ratio $S_p / S_s$ | **3.52** ← bad | **1.48** ← better | → <1.0 |

A ratio below 1.0 would indicate a representation organised primarily around seizure state. EEG-ViT moves from 3.52 to 1.48 (58% relative improvement).

---

## ⚠️ Limitations

- **Dataset scope:** CHB-MIT covers paediatric inpatients only. Validation on adult cohorts, ambulatory recordings, and multicentre data with heterogeneous hardware is needed before any deployment claim.
- **Distribution assumption:** LOSO assumes training and test subjects are drawn from the same clinical distribution. This may not hold across hospital sites.
- **Window isolation:** The ViT encoder processes each 5-second window independently. It cannot integrate evidence across consecutive windows — the root cause of perictal residual errors (the 389 remaining FPs and 45 FNs cluster at seizure boundaries).
- **Hyperparameter sensitivity:** $\lambda = 0.1$ (disentanglement strength) was selected on a held-out validation fold. Per-site tuning may be required in production.

---

## 🔭 Future Directions

| Direction | Motivation |
|-----------|-----------|
| Few-shot patient adaptation | Use a small calibration set at deployment to fine-tune for a new patient |
| Hierarchical temporal modelling | Stack a second attention layer across consecutive windows to resolve boundary errors |
| Supervised contrastive learning | Explicitly cluster ictal windows cross-patient via contrastive objectives |
| Graph neural networks | Model seizure propagation across the electrode adjacency graph |
| Multicentre validation | Test on adult EEG datasets with heterogeneous amplifier hardware |
| Neuromorphic deployment | Real-time embedded implementation on low-power edge hardware |
| Multimodal fusion | Combine EEG with wrist accelerometry for ambulatory monitoring |

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{vaishnavi2024eegvit,
  title     = {Bridging the Generalization Gap in EEG Seizure Detection:
               A Hybrid Vision Transformer Approach with Adversarial
               Subject Disentanglement},
  author    = {Vaishnavi, C and Rai, Yatindra and Thanigaivelu, P.S.},
  booktitle = {Proceedings of [Conference Name]},
  year      = {2024},
  institution = {SRM Institute of Science and Technology}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built at the Department of Computing Technologies, SRMIST*  
*Questions? Open an issue or reach out at yr5100@srmist.edu.in*

</div>
