import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# =====================================================
# PATHS
# =====================================================

FEATURE_PATH = "dataset/fusion_features.npy"
MODEL_PATH = "models/classifier.keras"
IMAGE_FOLDER = "dataset/processed_images"
OUTPUT_FOLDER = "evaluation"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("Loading Fusion Features...")
print("=" * 60)

# =====================================================
# LOAD FEATURES
# =====================================================

X = np.load(FEATURE_PATH)

print("Fusion Features :", X.shape)

# =====================================================
# CREATE LABELS FROM IMAGE FILENAMES
# =====================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

labels = []

for file in image_files:

    if file.startswith("img_1"):
        labels.append(1)

    elif file.startswith("img_2"):
        labels.append(0)

labels = np.array(labels)

N = min(len(X), len(labels))

X = X[:N]
y = labels[:N]

print("Samples :", N)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# LOAD MODEL
# =====================================================

print()
print("=" * 60)
print("Loading Classifier...")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Classifier Loaded Successfully")

# =====================================================
# PREDICTION
# =====================================================

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = (probabilities >= 0.5).astype(int).flatten()

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

# =====================================================
# PRINT RESULTS
# =====================================================

print()
print("=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =====================================================
# SAVE METRICS
# =====================================================

with open(
    os.path.join(OUTPUT_FOLDER, "metrics.txt"),
    "w"
) as f:

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n")

# =====================================================
# SAVE CLASSIFICATION REPORT
# =====================================================

with open(
    os.path.join(OUTPUT_FOLDER, "classification_report.txt"),
    "w"
) as f:

    f.write(report)

# =====================================================
# SAVE CONFUSION MATRIX
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Anemic"
    ]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.png"
    )
)

plt.close()

# =====================================================
# FINISHED
# =====================================================

print()
print("=" * 60)
print("Evaluation Completed Successfully")
print("=" * 60)

print("Saved Files")

print("evaluation/metrics.txt")
print("evaluation/classification_report.txt")
print("evaluation/confusion_matrix.png")