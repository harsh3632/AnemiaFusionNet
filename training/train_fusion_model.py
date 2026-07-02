import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.image_encoder import build_image_encoder
from models.clinical_encoder import build_clinical_encoder
from models.geo_encoder import build_geo_encoder
from models.transformer_fusion import build_transformer_fusion

print("="*60)
print("Loading Datasets...")
print("="*60)

# -------------------------------------------------------
# IMAGE DATA
# -------------------------------------------------------

IMAGE_FOLDER = "dataset/processed_images"

images = []

for file in os.listdir(IMAGE_FOLDER):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        img = cv2.imread(os.path.join(IMAGE_FOLDER, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        images.append(img)

images = np.array(images)

# -------------------------------------------------------
# CLINICAL DATA
# -------------------------------------------------------

clinical_df = pd.read_csv(
    "dataset/clinical_dataset/IDA_dataset_clean.csv"
)

clinical_df["Gender"] = clinical_df["Gender"].map(
    {"Male":1,"Female":0}
)

clinical_features = clinical_df[
    ["Hemoglobin","RDW","MCV","Age","Gender"]
].values.astype(np.float32)

# -------------------------------------------------------
# GEO DATA
# -------------------------------------------------------

geo_df = pd.read_csv(
    "dataset/geo_dataset/geo_risk_data.csv"
)

geo_df["Geo_Risk"] = geo_df["Geo_Risk"].map(
    {"Low":0,"Medium":1,"High":2}
)

geo_features = geo_df[
    ["Geo_Risk"]
].values.astype(np.float32)

# -------------------------------------------------------
# SAME SAMPLE SIZE
# -------------------------------------------------------

N = min(
    len(images),
    len(clinical_features),
    len(geo_features)
)

images = images[:N]
clinical_features = clinical_features[:N]
geo_features = geo_features[:N]

print(f"\nCommon Samples : {N}")

# -------------------------------------------------------
# LOAD TRAINED ENCODERS
# -------------------------------------------------------

image_encoder = tf.keras.models.load_model(
    "models/image_encoder.keras",
    compile=False
)

clinical_encoder = tf.keras.models.load_model(
    "models/clinical_encoder.keras",
    compile=False
)

geo_encoder = tf.keras.models.load_model(
    "models/geo_encoder.keras",
    compile=False
)

print("\nExtracting Features...")

image_feat = image_encoder.predict(images, verbose=0)
clinical_feat = clinical_encoder.predict(clinical_features, verbose=0)
geo_feat = geo_encoder.predict(geo_features, verbose=0)

print("Image Features    :", image_feat.shape)
print("Clinical Features :", clinical_feat.shape)
print("Geo Features      :", geo_feat.shape)
print("\nFeature Shapes")
print(image_feat.shape)
print(clinical_feat.shape)
print(geo_feat.shape)

assert image_feat.shape[1] == 128
assert clinical_feat.shape[1] == 128
assert geo_feat.shape[1] == 128
# -------------------------------------------------------
# FUSION MODEL
# -------------------------------------------------------

fusion_model = build_transformer_fusion()

fusion_model.compile(
    optimizer="adam",
    loss="mse"
)

dummy_target = np.zeros((N,128), dtype=np.float32)

X_train_img, X_test_img,\
X_train_clin, X_test_clin,\
X_train_geo, X_test_geo,\
y_train, y_test = train_test_split(
    image_feat,
    clinical_feat,
    geo_feat,
    dummy_target,
    test_size=0.2,
    random_state=42
)

print("\nTraining Fusion Model...")

fusion_model.fit(
    [
        X_train_img,
        X_train_clin,
        X_train_geo
    ],
    y_train,
    validation_data=(
        [
            X_test_img,
            X_test_clin,
            X_test_geo
        ],
        y_test
    ),
    epochs=15,
    batch_size=16
)

print("\nGenerating Fusion Features...")

fusion_features = fusion_model.predict(
    [
        image_feat,
        clinical_feat,
        geo_feat
    ],
    verbose=0
)

os.makedirs("dataset", exist_ok=True)

np.save(
    "dataset/fusion_features.npy",
    fusion_features
)

fusion_model.save(
    "models/fusion_model.keras"
)

print("="*60)
print("Fusion Training Completed")
print("="*60)
print("Fusion Feature Shape :", fusion_features.shape)
print("Saved : models/fusion_model.keras")
print("Saved : dataset/fusion_features.npy")