# 🩸 AnemiaFusionNet

## Transformer-Based Multimodal Anemia Detection Framework

AnemiaFusionNet is a deep learning framework designed for anemia prediction using multimodal healthcare data. The system combines three different data modalities:

- Eye Conjunctiva Images
- Clinical Blood Parameters
- Geographic Risk Information

Each modality is processed using an independent encoder. The extracted features are fused through a Transformer-based fusion network before final anemia classification.

The project is developed as an academic implementation of a multimodal artificial intelligence framework for anemia detection.

---

# Project Objectives

The main objectives of this project are:

- Detect anemia using multimodal data.
- Extract image features using EfficientNetB0.
- Learn clinical representations using a neural encoder.
- Encode geographic anemia risk information.
- Fuse multimodal features using a Transformer Network.
- Predict anemia using a deep learning classifier.
- Provide an easy-to-use Streamlit web application.

---

# Key Features

- Multimodal AI Framework
- Eye Image Analysis
- Clinical Data Analysis
- Geographic Risk Integration
- Transformer-based Feature Fusion
- Binary Anemia Classification
- Streamlit Web Application
- End-to-End Deep Learning Pipeline

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Image Backbone | EfficientNetB0 |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |
| Web Application | Streamlit |

---

# Project Highlights

- Independent Image Encoder
- Independent Clinical Encoder
- Independent Geo Encoder
- Transformer-based Multimodal Fusion
- Binary Deep Learning Classifier
- Interactive Streamlit Interface
- Performance Evaluation Module

---
# Project Structure

```text
AnemiaFusionNet/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── image_dataset/
│   ├── clinical_dataset/
│   ├── geo_dataset/
│   ├── processed_images/
│   └── fusion_features.npy
│
├── preprocessing/
│   ├── image_preprocessing.py
│   ├── clinical_preprocessing.py
│   ├── geo_preprocessing.py
│   ├── geo_risk_assignment.py
│   └── merge_data.py
│
├── models/
│   ├── image_encoder.py
│   ├── clinical_encoder.py
│   ├── geo_encoder.py
│   ├── transformer_fusion.py
│   ├── classifier.py
│   ├── image_encoder.keras
│   ├── clinical_encoder.keras
│   ├── geo_encoder.keras
│   ├── fusion_model.keras
│   └── classifier.keras
│
├── training/
│   ├── train_image_encoder.py
│   ├── train_clinical_encoder.py
│   ├── train_geo_encoder.py
│   ├── train_fusion_model.py
│   └── train_classifier.py
│
├── utils/
│   ├── image_utils.py
│   ├── clinical_utils.py
│   ├── geo_utils.py
│   └── prediction.py
│
├── evaluation/
│   ├── evaluate_classifier.py
│   ├── metrics.txt
│   ├── classification_report.txt
│   └── confusion_matrix.png
│
└── outputs/
```

---

# Dataset

The project combines three independent datasets for multimodal anemia prediction.

## 1. Eye Image Dataset

**Purpose**

Eye conjunctiva images are used for visual feature extraction.

**Processed Samples**

- 183 Images

**Preprocessing**

- Resize to 224 × 224
- RGB Conversion
- Normalization

---

## 2. Clinical Dataset

Clinical attributes include:

- Hemoglobin
- RDW
- MCV
- Age
- Gender

**Samples**

- 1000 Records

---

## 3. Geographic Dataset

The geographical dataset provides district-level anemia risk.

Clinical information is combined with regional risk to improve prediction.

**Samples**

- 707 District Records

---

# Data Preprocessing

The preprocessing pipeline consists of:

### Image Preprocessing

- Image Loading
- Image Resizing
- RGB Conversion
- Normalization

### Clinical Preprocessing

- Missing Value Handling
- Gender Encoding
- Feature Selection

### Geographic Preprocessing

- District Cleaning
- Risk Assignment
- Geo Risk Encoding

---

# Project Workflow

The complete workflow is:

```text
Eye Image
      │
      ▼
Image Encoder
      │
      ▼
Clinical Encoder
      │
      ▼
Geo Encoder
      │
      ▼
Transformer Fusion
      │
      ▼
Classifier
      │
      ▼
Final Prediction
```

---
# Model Architecture

The proposed framework follows a multimodal deep learning architecture where each data modality is processed independently before feature fusion.

## Image Encoder

The eye conjunctiva image is processed using **EfficientNetB0** as the backbone network.

Feature extraction pipeline:

- Input Size: 224 × 224 × 3
- EfficientNetB0 Backbone
- Global Average Pooling
- Dense Layer
- Dropout
- 128-Dimensional Feature Vector

---

## Clinical Encoder

Clinical parameters are passed through a fully connected neural encoder.

Input Features:

- Hemoglobin
- RDW
- MCV
- Age
- Gender

Output:

- 128-Dimensional Clinical Feature Vector

---

## Geographic Encoder

The geographic encoder processes district-level anemia risk.

Input:

- Geo Risk

Output:

- 128-Dimensional Geographic Feature Vector

---

## Transformer Fusion Network

The three feature vectors are fused using a Transformer-based fusion module.

Fusion Inputs:

- Image Feature Vector
- Clinical Feature Vector
- Geographic Feature Vector

Transformer Components:

- Multi-Head Self Attention
- Feed Forward Network
- Layer Normalization
- Global Average Pooling

Output:

- 128-Dimensional Fusion Feature Vector

---

## Final Classifier

The fused representation is passed to a binary classifier.

Output Classes:

- Normal
- Anemic

---

# Training Pipeline

The training process consists of five independent stages.

## Stage 1

Train Image Encoder

↓

## Stage 2

Train Clinical Encoder

↓

## Stage 3

Train Geographic Encoder

↓

## Stage 4

Train Transformer Fusion Model

↓

## Stage 5

Train Final Binary Classifier

---

# Performance Evaluation

The classifier is evaluated using the following metrics.

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Generated evaluation files:

- metrics.txt
- classification_report.txt
- confusion_matrix.png

---

# Streamlit Application

The project includes an interactive Streamlit web application.

Available modules:

- Upload Eye Image
- Enter Clinical Parameters
- Select Geographic Risk
- Predict Anemia
- Display Prediction
- Show Prediction Confidence
- Display Model Information
- Display Dataset Information

The application integrates all trained deep learning models into a single interface for multimodal anemia prediction.

---
# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AnemiaFusionNet.git
```

Move to the project directory:

```bash
cd AnemiaFusionNet
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install all required packages:

```bash
pip install -r requirements.txt
```

---

# Project Execution

## Step 1

Run Image Preprocessing

```bash
python -m preprocessing.image_preprocessing
```

---

## Step 2

Run Clinical Preprocessing

```bash
python -m preprocessing.clinical_preprocessing
```

---

## Step 3

Run Geographic Preprocessing

```bash
python -m preprocessing.geo_preprocessing
```

---

## Step 4

Train Image Encoder

```bash
python -m training.train_image_encoder
```

---

## Step 5

Train Clinical Encoder

```bash
python -m training.train_clinical_encoder
```

---

## Step 6

Train Geographic Encoder

```bash
python -m training.train_geo_encoder
```

---

## Step 7

Train Transformer Fusion Model

```bash
python -m training.train_fusion_model
```

---

## Step 8

Train Final Classifier

```bash
python -m training.train_classifier
```

---

## Step 9

Evaluate the Model

```bash
python -m evaluation.evaluate_classifier
```

---

## Step 10

Run Streamlit Application

```bash
python -m streamlit run app.py
```

---

# Expected Output

The Streamlit application allows users to:

- Upload an eye conjunctiva image.
- Enter clinical parameters.
- Select geographic risk.
- Predict anemia.
- View prediction probability.
- View prediction confidence.
- Display project workflow.
- Display dataset information.

---

# Experimental Results

The current implementation produced the following evaluation results.

| Metric | Value |
|---------|------:|
| Accuracy | 54.05% |
| Precision | 0.00 |
| Recall | 0.00 |
| F1 Score | 0.00 |

> **Note:** The current performance is limited by the small image dataset (183 processed images) and the current training strategy. This project primarily demonstrates the implementation of a multimodal Transformer-based framework rather than achieving state-of-the-art diagnostic performance.

---

# Future Scope

Possible future improvements include:

- Increase the image dataset size.
- End-to-end multimodal training.
- Fine-tune the EfficientNetB0 backbone.
- Improve class balancing.
- Use advanced data augmentation.
- Add explainable AI (XAI) methods.
- Deploy the application to a cloud platform.

---
# References

The project implementation is based on publicly available datasets and widely used deep learning frameworks.

### Image Dataset

Detecting Anaemia Using Computer Vision and Machine Learning (Kaggle)

### Clinical Dataset

Iron Deficiency Anemia Clinical Dataset (Kaggle)

### Geographic Dataset

National Family Health Survey (NFHS-5), Government of India

### Deep Learning Frameworks

- TensorFlow
- Keras
- EfficientNetB0
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Streamlit

---

# Acknowledgements

This project was developed as an academic implementation of a Transformer-Based Multimodal Artificial Intelligence framework for anemia prediction.

Special thanks to:

- Kaggle Dataset Contributors
- TensorFlow Development Team
- Streamlit Development Team
- OpenCV Community
- Scikit-learn Developers

for providing publicly available datasets and open-source tools that supported this implementation.

---

# License

This repository is intended for academic and educational purposes.

The project should not be used for real-world medical diagnosis or clinical decision-making without proper medical validation.

---

# Repository Contents

The repository contains:

- Complete Source Code
- Dataset Preprocessing Pipeline
- Model Training Scripts
- Trained Deep Learning Models
- Evaluation Scripts
- Streamlit Application
- Documentation
- Performance Reports

---

# Project Summary

AnemiaFusionNet demonstrates a complete multimodal deep learning pipeline for anemia prediction.

The framework integrates:

- Eye Conjunctiva Images
- Clinical Blood Parameters
- Geographic Risk Information

The extracted representations are combined using a Transformer-based fusion model followed by a binary classifier for final prediction.

The project includes:

- Data Preprocessing
- Feature Extraction
- Transformer Fusion
- Model Training
- Performance Evaluation
- Streamlit Deployment

---

# Author

**Harsh Patel**

Academic Deep Learning Project

---

# Contact

For academic discussions or project-related queries, please contact the repository owner through GitHub.

---

# Version

**Current Version:** 1.0

**Status:** Completed

---

⭐ If you found this project useful, consider giving the repository a star.
