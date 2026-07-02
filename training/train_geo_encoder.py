import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from models.geo_encoder import build_geo_encoder

print("=" * 60)
print("Loading Geo Dataset...")
print("=" * 60)

df = pd.read_csv("dataset/geo_dataset/geo_risk_data.csv")

print(df.head())

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()

X = encoder.fit_transform(
    df["Geo_Risk"]
).reshape(-1, 1).astype(np.float32)

# Binary target for temporary training
y = (df["Geo_Risk"] == "High").astype(np.float32)

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

geo_encoder = build_geo_encoder()

# =====================================================
# TEMP CLASSIFIER
# =====================================================

x = geo_encoder.output

prediction = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

training_model = tf.keras.Model(
    geo_encoder.input,
    prediction
)

training_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "temp_geo.keras",
    monitor="val_accuracy",
    save_best_only=True
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

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

loss, acc = training_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("=" * 60)
print("Training Completed")
print("=" * 60)

print(f"Accuracy : {acc:.4f}")
print(f"Loss     : {loss:.4f}")

# =====================================================
# SAVE ONLY ENCODER
# =====================================================

geo_encoder.save("models/geo_encoder.keras")

print()
print("Encoder Saved Successfully")
print("models/geo_encoder.keras")