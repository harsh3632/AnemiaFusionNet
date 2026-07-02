import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from models.clinical_encoder import build_clinical_encoder

# =====================================================
# PATHS
# =====================================================

DATA_PATH = "dataset/clinical_dataset/IDA_dataset_clean.csv"
MODEL_PATH = "models/clinical_encoder.keras"

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("Loading Clinical Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(df.head())

# =====================================================
# LABEL ENCODING
# =====================================================

gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

target_encoder = LabelEncoder()
df["Anemia_Type"] = target_encoder.fit_transform(df["Anemia_Type"])

# =====================================================
# FEATURES
# =====================================================

X = df[
    [
        "Hemoglobin",
        "RDW",
        "MCV",
        "Age",
        "Gender"
    ]
]

y = df["Anemia_Type"]

# =====================================================
# NORMALIZATION
# =====================================================

scaler = StandardScaler()

X = scaler.fit_transform(X).astype(np.float32)

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

print()
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# BUILD ENCODER
# =====================================================

encoder = build_clinical_encoder()

# =====================================================
# TEMP CLASSIFIER
# =====================================================

x = encoder.output

prediction = tf.keras.layers.Dense(
    len(np.unique(y)),
    activation="softmax"
)(x)

training_model = tf.keras.Model(
    encoder.input,
    prediction
)

training_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "temp_clinical.keras",
    monitor="val_accuracy",
    save_best_only=True
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =====================================================
# TRAIN
# =====================================================

print()
print("=" * 60)
print("Training Started...")
print("=" * 60)

training_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

loss, accuracy = training_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("=" * 60)
print("Training Completed")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Loss     : {loss:.4f}")

# =====================================================
# SAVE ONLY ENCODER
# =====================================================

encoder.save(MODEL_PATH)

print()
print("Encoder Saved Successfully")
print(MODEL_PATH)