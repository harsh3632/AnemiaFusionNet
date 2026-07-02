import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from models.image_encoder import build_image_encoder

# =====================================================
# PATHS
# =====================================================

IMAGE_FOLDER = "dataset/processed_images"
MODEL_SAVE_PATH = "models/image_encoder.keras"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16
EPOCHS = 20

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("Loading Image Dataset...")
print("=" * 60)

images = []
labels = []

for file in sorted(os.listdir(IMAGE_FOLDER)):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(IMAGE_FOLDER, file)

        image = load_img(path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = image / 255.0

        images.append(image)

        # Dummy labels (replace with real labels later)
        if len(labels) % 2 == 0:
            labels.append(0)
        else:
            labels.append(1)

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.float32)

print("Images :", len(images))
print("Labels :", len(labels))

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print()
print("Training Images :", len(X_train))
print("Testing Images  :", len(X_test))

# =====================================================
# BUILD ENCODER
# =====================================================

encoder = build_image_encoder()

# =====================================================
# TEMP CLASSIFIER (ONLY FOR TRAINING)
# =====================================================

x = encoder.output

prediction = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

training_model = tf.keras.Model(
    encoder.input,
    prediction
)

training_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "temp_image.keras",
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
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, earlystop]
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

encoder.save(MODEL_SAVE_PATH)

print()
print("Encoder Saved Successfully")
print(MODEL_SAVE_PATH)