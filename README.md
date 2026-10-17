# Emotion Recognition CNN

> 3-layer convolutional neural network for 7-class facial emotion classification. Built and benchmarked against classical and modern baselines.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg)](https://tensorflow.org/)

---

## What it does

Classifies facial expressions into 7 emotion categories: anger, contempt, disgust, fear, happiness, sadness, surprise.

Trained on the [CK+ (Extended Cohn-Kanade) dataset](https://www.jeffcohn.net/Resources/).

---

## Results

| Model | Test accuracy | Weighted F1 |
|---|---|---|
| **Custom CNN (ours)** | **97.0%** | **0.97** |
| MobileNetV2 (transfer) | 94.5% | 0.94 |
| MLP | 82.1% | 0.81 |
| SVM | 79.4% | 0.78 |
| KNN | 71.2% | 0.70 |

### Per-class breakdown (CNN)

| Emotion | Precision | Recall |
|---|---|---|
| Anger | 1.00 | 0.95 |
| Contempt | 1.00 | 1.00 |
| Disgust | 0.95 | 1.00 |
| **Fear** | **1.00** | **0.85** |
| Happiness | 0.95 | 1.00 |
| Sadness | 0.95 | 1.00 |
| Surprise | 1.00 | 0.98 |

Fear class underperforms on recall (0.85) despite perfect precision. Likely cause: CK+ has fewer fear samples than other classes, creating a data imbalance. A targeted augmentation or class-weighted loss would address this.

---

## Architecture

```
Input (48 × 48 × 1 grayscale)
      │
      ▼
Conv2D(32) → ReLU → MaxPool
      │
      ▼
Conv2D(64) → ReLU → MaxPool
      │
      ▼
Conv2D(128) → ReLU → MaxPool
      │
      ▼
Flatten → Dropout(0.5)
      │
      ▼
Dense(128) → ReLU
      │
      ▼
Dense(7) → Softmax
```

---

## Tech stack

- Python 3.10+
- TensorFlow / Keras
- OpenCV (image preprocessing)
- scikit-learn (baselines, metrics)
- NumPy, pandas, Matplotlib

---

## Setup

```bash
git clone https://github.com/MohanadddAmrr/Emotion-Recognition-CNN.git
cd Emotion-Recognition-CNN

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Download the CK+ dataset and place it under `data/raw/CK+`.

```bash
python src/train.py
python src/evaluate.py
```

---

## Context

Academic project for the Deep Learning course at Misr International University (MIU), BSc Data Science.

---

## License

MIT
