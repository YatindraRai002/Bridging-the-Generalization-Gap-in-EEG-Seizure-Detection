# Bridging the Generalization Gap in EEG Seizure Detection: A Hybrid Vision Transformer with Adversarial Subject Disentanglement

[![Paper](https://img.shields.io/badge/Paper-IJDDT%202026-blue)](http://doi.org/10.25258/ijddt.16.55s.19)
[![Research](https://img.shields.io/badge/Domain-Healthcare%20AI-green)]()
[![Task](https://img.shields.io/badge/Task-EEG%20Seizure%20Detection-orange)]()
[![Model](https://img.shields.io/badge/Model-Hybrid%20EEG--ViT-red)]()

## 📖 Overview

This repository accompanies our published research paper:

**Bridging the Generalization Gap in EEG Seizure Detection: A Hybrid Vision Transformer with Adversarial Subject Disentanglement**

Published in **International Journal of Drug Delivery Technology (IJDDT), Volume 16, Issue 55s, 2026**.

🔗 **DOI:** http://doi.org/10.25258/ijddt.16.55s.19

---

## 📌 Abstract

Automated EEG-based seizure detection systems often achieve high performance when evaluated on subjects seen during training but fail to generalize to unseen patients. This limitation arises because EEG signals contain strong subject-specific characteristics that models inadvertently learn instead of seizure-related patterns.

To address this challenge, we propose a **Hybrid EEG Vision Transformer (EEG-ViT)** that combines:

* Depthwise Spatial Convolutions
* Multi-Head Self Attention
* Focal Loss for Class Imbalance
* Adversarial Subject Disentanglement using Gradient Reversal

Our framework explicitly encourages the model to learn seizure-discriminative representations while suppressing patient identity information, leading to significantly improved cross-subject performance under a Leave-One-Subject-Out (LOSO) evaluation protocol.

---

# 🎯 Research Motivation

Most seizure detection models report impressive performance under patient-dependent evaluation settings.

However, real-world deployment requires models that can generalize across unseen individuals.

The key research question addressed in this work:

> Can we learn seizure-relevant EEG representations while preventing the model from memorizing patient-specific characteristics?

---

# 🏗️ Proposed Architecture

```text
Raw EEG Signals
        │
        ▼
Depthwise Spatial Convolution
        │
        ▼
Temporal Convolution Blocks
        │
        ▼
Patch Embedding
        │
        ▼
Vision Transformer Encoder
        │
        ▼
Shared Latent Representation
        │
 ┌──────┴──────┐
 ▼             ▼
Seizure Head   Patient Identity Head
(Focal Loss)   (GRL + Adversarial Loss)
```

The model utilizes a Gradient Reversal Layer (GRL) to enforce subject-invariant feature learning while preserving seizure-related information.

---

# 🔬 Methodology

## Dataset

### CHB-MIT Scalp EEG Dataset

* 22 Patients
* 198 Annotated Seizure Events
* 23 EEG Channels
* 256 Hz Sampling Rate

---

## Evaluation Protocol

### Leave-One-Subject-Out (LOSO)

For each fold:

* Train on 21 patients
* Test on 1 unseen patient

This protocol provides a realistic assessment of model generalization.

---

## Key Components

### Hybrid EEG-ViT

Combines:

* CNN-based local feature extraction
* Transformer-based global context modeling

---

### Focal Loss

Addresses severe class imbalance between:

* Seizure windows
* Non-seizure windows

---

### Adversarial Subject Disentanglement

Uses:

* Patient Identity Classifier
* Gradient Reversal Layer (GRL)

to suppress patient-specific information.

---

# 📊 Results

## Patient 7 (Most Challenging Subject)

| Metric      | Baseline CNN | EEG-ViT |
| ----------- | ------------ | ------- |
| Accuracy    | 65.55%       | 87.67%  |
| F1 Score    | 0.1035       | 0.5220  |
| Sensitivity | 24.82%       | 84.04%  |
| Specificity | 69.10%       | 87.99%  |

### Key Improvements

✅ +22.12% Accuracy

✅ +59.22% Sensitivity

✅ 8.7× Reduction in False Alarm Burden

✅ Significant Improvement in Cross-Subject Generalization

---

# 📈 Representation Analysis

To understand *why* performance improved, we analyzed latent representations using Silhouette Scores.

### Baseline CNN

```text
Patient Separation = 0.74
Seizure Separation = 0.21
Ratio = 3.52
```

The model primarily learned patient identity.

---

### Proposed EEG-ViT

```text
Patient Separation = 0.57
Seizure Separation = 0.39
Ratio = 1.48
```

The learned representation became substantially more seizure-centric and subject-invariant.

---

# 📚 Publication

### Citation

```bibtex
@article{vaishnavi2026eegvit,
  title={Bridging the Generalization Gap in EEG Seizure Detection: A Hybrid Vision Transformer with Adversarial Subject Disentanglement},
  author={Vaishnavi, C and Rai, Yatindra and Thanigaivelu, P.S.},
  journal={International Journal of Drug Delivery Technology},
  volume={16},
  number={55s},
  pages={172--178},
  year={2026},
  doi={10.25258/ijddt.16.55s.19}
}
```

---

# 👥 Authors

### C. Vaishnavi

Department of Computing Technologies
SRM Institute of Science and Technology

### Yatindra Rai

Department of Computing Technologies
SRM Institute of Science and Technology

### Dr. Thanigaivelu P. S.

Department of Computing Technologies
SRM Institute of Science and Technology

---

# 🙏 Acknowledgements

We would like to thank the Department of Computing Technologies, SRM Institute of Science and Technology, for supporting this research.

We also acknowledge the CHB-MIT Scalp EEG Database for providing the benchmark dataset used in this study.

---

# ⭐ If you find this work useful

Please consider starring the repository and citing our paper.

Research advances when knowledge is shared.
