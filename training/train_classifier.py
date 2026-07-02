import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.classifier import build_classifier

print("=" * 60)
print("Loading Fusion Features...")
print("=" * 60)

# =====================================================
# LOAD FEATURES
# =====================================================

FEATURE_PATH = "dataset/fusion_features.npy"

fusion_features = np.load(FEATURE_PATH)

print("Fusion Features :", fusion_features.shape)

# =====================================================
# CREATE LABELS
# (Temporary labels based on image naming rule)
# =====================================================

IMAGE_FOLDER = "dataset/processed_images"

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

N = min(len(fusion_features), len(labels))

fusion_features = fusion_features[:N]
labels = labels[:N]

print("Samples :", N)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    fusion_features,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print()
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# BUILD MODEL
# =====================================================

model = build_classifier()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# CALLBACKS
# =====================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/classifier.keras",
    monitor="val_accuracy",
    save_best_only=True
)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =====================================================
# TRAIN
# =====================================================

print()
print("=" * 60)
print("Training Classifier...")
print("=" * 60)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

# =====================================================
# EVALUATE
# =====================================================

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("=" * 60)
print("Classifier Training Completed")
print("=" * 60)

print(f"Accuracy : {acc:.4f}")
print(f"Loss     : {loss:.4f}")

print()
print("Saved Successfully")
print("models/classifier.keras")