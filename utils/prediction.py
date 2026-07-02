import os
import sys
import tensorflow as tf

# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# =====================================================
# IMPORT CUSTOM LAYER
# =====================================================

from models.transformer_fusion import TransformerBlock

# =====================================================
# LOAD TRAINED MODELS
# =====================================================

print("=" * 60)
print("Loading Trained Models...")
print("=" * 60)

image_encoder = tf.keras.models.load_model(
    "models/image_encoder.keras",
    compile=False
)
print("✓ Image Encoder Loaded")

clinical_encoder = tf.keras.models.load_model(
    "models/clinical_encoder.keras",
    compile=False
)
print("✓ Clinical Encoder Loaded")

geo_encoder = tf.keras.models.load_model(
    "models/geo_encoder.keras",
    compile=False
)
print("✓ Geo Encoder Loaded")

fusion_model = tf.keras.models.load_model(
    "models/fusion_model.keras",
    custom_objects={
        "TransformerBlock": TransformerBlock
    },
    compile=False
)
print("✓ Fusion Model Loaded")

classifier = tf.keras.models.load_model(
    "models/classifier.keras",
    compile=False
)
print("✓ Classifier Loaded")

print("=" * 60)
print("All Models Loaded Successfully")
print("=" * 60)
import numpy as np

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(
    image,
    clinical_data,
    geo_data
):

    # Image Feature
    image_feature = image_encoder.predict(
        image,
        verbose=0
    )

    # Clinical Feature
    clinical_feature = clinical_encoder.predict(
        clinical_data,
        verbose=0
    )

    # Geo Feature
    geo_feature = geo_encoder.predict(
        geo_data,
        verbose=0
    )

    return (
        image_feature,
        clinical_feature,
        geo_feature
    )


# =====================================================
# FUSION FEATURE
# =====================================================

def generate_fusion_feature(

    image_feature,
    clinical_feature,
    geo_feature

):

    fusion_feature = fusion_model.predict(

        [
            image_feature,
            clinical_feature,
            geo_feature
        ],

        verbose=0

    )

    return fusion_feature
# =====================================================
# FINAL PREDICTION
# =====================================================

def predict_anemia(
    image,
    clinical_data,
    geo_data
):

    # Extract modality features
    image_feature, clinical_feature, geo_feature = extract_features(
        image,
        clinical_data,
        geo_data
    )

    # Generate fusion feature
    fusion_feature = generate_fusion_feature(
        image_feature,
        clinical_feature,
        geo_feature
    )

    # Final prediction
    probability = classifier.predict(
        fusion_feature,
        verbose=0
    )[0][0]

    prediction = "Anemic" if probability >= 0.5 else "Normal"

    confidence = probability if probability >= 0.5 else (1.0 - probability)

    return {
        "prediction": prediction,
        "probability": float(probability),
        "confidence": float(confidence)
    }


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    print("\nPrediction Module Ready Successfully.")