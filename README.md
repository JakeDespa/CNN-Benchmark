# 🌿 Vision Benchmark & Ensemble Studio

A comprehensive comparative study and benchmarking framework evaluating **17 Convolutional Neural Network (CNN) architectures** and **45 pairwise soft-voting ensembles** for image classification across both high-performance **GPU** and resource-constrained **Low-Spec CPU** hardware profiles.

---

## 📌 Project Overview

This repository contains the training scripts, benchmark results, evaluation curves, deduplication audit tools, and an interactive **Streamlit dashboard** for fine-tuning pretrained vision models on a 3-class recyclable waste dataset (`metal`, `paper`, `plastic`).

### Core Highlights:
- **17 CNN Models Benchmarked:** AlexNet, VGG-11, GoogLeNet, ResNet-18, Inception-v3, SqueezeNet 1.1, Wide ResNet-50-2, DenseNet-121, ResNeXt-50 32x4d, MobileNetV2, ShuffleNetV2 x0.5, MNASNet 0.5, EfficientNet-B0, MobileNetV3-Small, RegNetY-400MF, EfficientNetV2-S, ConvNeXt-Tiny.
- **45 Pairwise Ensembles:** All $\binom{10}{2} = 45$ combinations evaluated via soft-voting probability averaging ($\operatorname{argmax}(\frac{P_A + P_B}{2})$).
- **Dual Hardware Execution Profiles:**
  - 🚀 **Full GPU Profile:** CUDA acceleration, Automatic Mixed Precision (AMP), batch size 32.
  - 💻 **Low-Spec CPU Profile:** Strictly CPU-only, hard-capped to 2 CPU threads (`torch.set_num_threads(2)`), and optimized for a 4 GB RAM footprint (simulating edge single-board computers like the Raspberry Pi 4).
- **Zero Dataset Leakage:** Validated with MD5 checksums and perceptual difference hashing (dHash $\le 4$), confirming 0 cross-split duplicates across all 1,625 images.
- **Interactive UI Dashboard (`app.py`):** Explore leaderboards, compare pairwise ensembles, inspect confusion matrices, and test real-time image classification.

---

## 🗂️ Repository Structure

```text
├── app.py                     # Interactive Streamlit Dashboard
├── dupe.py                    # Dataset duplicate & leakage audit tool
├── README.md                  # Documentation
├── .gitignore                 # Excludes large binaries (>100MB) & datasets
│
├── newtrain/                  # GPU Training Benchmarks
│   ├── indiv_aug/             # 17 models with training data augmentation
│   ├── indiv_noaug/           # 17 models without data augmentation
│   └── ensemble_aug/          # 10 base models + 45 soft-voting ensembles
│       └── results/           # CSVs, reports, and summary plots
│
├── limited/                   # Low-Spec CPU Profile (2 Threads, 4 GB RAM)
│   ├── indiv_aug/             # Script & Notebook (.py / .ipynb)
│   ├── indiv_noaug/           # Script & Notebook (.py / .ipynb)
│   └── ensemble_aug/          # Script & Notebook (.py / .ipynb)
│       └── results/           # Isolated benchmark outputs
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install torch torchvision pandas numpy pillow matplotlib streamlit scikit-learn
```

### 2. Launch the Interactive Dashboard

Launch the Streamlit UI to inspect model metrics, pairwise synergy gains, and test live predictions:

```bash
streamlit run app.py
```

### 3. Run Benchmark Scripts Locally

To execute any benchmark locally on your machine:

```bash
# Example: Run Low-Spec CPU ensemble benchmark
cd limited/ensemble_aug
python run_ensemble_aug.py
```

---

## 📊 Recorded Benchmark Metrics

For every model and ensemble combination, the following metrics are recorded under `./results/results.csv`:
- **Classification Performance:** Test Accuracy, Macro/Weighted Precision, Recall, and F1-score, Test Loss.
- **Hardware Efficiency:** Training Time, Inference Latency (batch size = 1, FP32, in milliseconds), Parameter Count (Millions), and Model Size (MB).
- **Ensemble Dynamics:** Accuracy gain over best constituent member, joint inference latency, and agreement rate.

---
