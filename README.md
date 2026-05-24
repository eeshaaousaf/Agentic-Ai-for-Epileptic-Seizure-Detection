<div align="center">

# 🧠 Agentic AI for Epileptic Seizure Detection

### Intelligent EEG-Based Seizure Detection using Machine Learning & Deep Learning

<img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow" />
<img src="https://img.shields.io/badge/Scikit--Learn-ML-red?style=for-the-badge&logo=scikitlearn" />
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
<img src="https://img.shields.io/badge/Domain-Healthcare_AI-purple?style=for-the-badge" />

</div>

---

#  Overview

Epilepsy is a serious neurological disorder characterized by abnormal electrical activity in the brain, leading to unpredictable seizures. Manual EEG analysis is time-consuming, complex, and prone to human error.

This project presents an **Agentic AI-based automated seizure detection system** that analyzes EEG signals and classifies them into:

- **Seizure (Ictal)**
- **Non-Seizure (Inter-Ictal)**

The system leverages both:

-  **Random Forest Classifier** (Machine Learning)
-  **Convolutional Neural Network (CNN)** (Deep Learning)

to compare performance and evaluate the effectiveness of AI in biomedical signal classification.

---

# Objectives

- Build an end-to-end automated seizure detection pipeline
- Process raw EEG EDF files using AI techniques
- Perform signal preprocessing and feature extraction
- Handle dataset imbalance using SMOTE
- Train and compare ML and DL models
- Evaluate model performance using classification metrics
- Explore AI applications in intelligent healthcare systems

---

#  Dataset

## CHB-MIT Scalp EEG Database

This project uses the publicly available **CHB-MIT EEG Dataset** from PhysioNet.

### Dataset Statistics

| Parameter | Value |
|---|---|
| Total Patients | 17 |
| EDF Files | 247 |
| EEG Channels | 23 |
| Sampling Rate | 256 Hz |
| Processed Files | 78 |
| Final Dataset | 36 Samples × 139 Features |

---

#  Project Pipeline

```text
EDF EEG Files
      ↓
Signal Extraction
      ↓
Preprocessing & Normalization
      ↓
Feature Extraction
      ↓
SMOTE Class Balancing
      ↓
Train/Test Split
      ↓
Random Forest + CNN Training
      ↓
Evaluation & Prediction
```

---

#  Data Preprocessing

The preprocessing pipeline includes:

- EDF file parsing using MNE
- EEG signal extraction
- Amplitude normalization
- Artifact removal
- Annotation-based labeling
- Missing value verification
- Duplicate removal
- Outlier detection using IQR

---

#  Exploratory Data Analysis (EDA)

## Dataset Cleaning

| Step | Shape |
|---|---|
| Initial Feature Dataset | (78, 139) |
| After Outlier Removal | (36, 139) |

### Class Distribution Before SMOTE

| Class | Samples |
|---|---|
| Non-Seizure | 22 |
| Seizure | 14 |

### Class Distribution After SMOTE

| Class | Samples |
|---|---|
| Non-Seizure | 22 |
| Seizure | 22 |

---

#  Feature Extraction

For each of the 23 EEG channels, the following statistical features were extracted:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Median
- Variance

### Total Features

```text
23 Channels × 6 Features = 138 Features
+ 1 Label Column
= 139 Total Columns
```

---

#  Models Used

## 1️⃣ Random Forest Classifier

A machine learning baseline model trained using Scikit-Learn.

### Libraries

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
```

### Performance

| Metric | Score |
|---|---|
| Accuracy | 62.5% |
| Precision | 0.50 |
| Recall | 0.33 |
| F1-Score | 0.40 |
| AUC-ROC | 0.50 |

---

## 2️⃣ Convolutional Neural Network (CNN)

A deep learning model developed using TensorFlow/Keras.

### Architecture

- Conv1D Layers
- Batch Normalization
- MaxPooling
- Dropout
- Dense Layers
- Sigmoid Output Layer

### Libraries

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
```

### Performance

| Metric | Score |
|---|---|
| Accuracy | 87.5% |
| Precision | 1.00 |
| Recall | 0.67 |
| F1-Score | 0.80 |
| AUC-ROC | 0.73 |

---

#  Results Comparison

| Metric | Random Forest | CNN |
|---|---|---|
| Accuracy | 62.5% | 87.5% |
| Precision | 0.50 | 1.00 |
| Recall | 0.33 | 0.67 |
| F1-Score | 0.40 | 0.80 |
| AUC-ROC | 0.50 | 0.73 |

---

#  Key Findings

✅ CNN significantly outperformed Random Forest

✅ Deep Learning captured temporal EEG patterns more effectively

✅ Perfect seizure precision achieved by CNN

✅ Demonstrated strong potential for intelligent healthcare systems

---

#  Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| ML Framework | Scikit-Learn |
| DL Framework | TensorFlow / Keras |
| EEG Processing | MNE |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Imbalanced Learning | SMOTE |
| Dataset Source | PhysioNet CHB-MIT |

---

#  Project Structure

```bash
Agentic-AI-Seizure-Detection/
│
├── data/
│   ├── raw_edf_files/
│   └── processed_dataset.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── models/
│   ├── random_forest.pkl
│   └── cnn_model.h5
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_rf.py
│   └── train_cnn.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_graph.png
│   └── roc_curve.png
│
├── requirements.txt
├── README.md
└── report.pdf
```

---

#  Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Agentic-AI-Seizure-Detection.git
cd Agentic-AI-Seizure-Detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

#  Usage

## Train Random Forest

```bash
python train_rf.py
```

## Train CNN

```bash
python train_cnn.py
```


# Applications

- Smart Healthcare Systems
- Clinical Decision Support
- Neurological Monitoring
- Wearable AI Devices
- Real-Time Seizure Alert Systems

---


#  License

This project is intended for academic and research purposes.

---

<div align="center">


</div>
