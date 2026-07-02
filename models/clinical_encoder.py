import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# CLINICAL ENCODER
# =====================================================

INPUT_FEATURES = 5

def build_clinical_encoder():

    inputs = tf.keras.Input(
        shape=(INPUT_FEATURES,),
        name="clinical_input"
    )

    x = layers.Dense(
        64,
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.30)(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="clinical_features"
    )(x)

    model = models.Model(
        inputs,
        outputs=x,
        name="Clinical_Encoder"
    )

    return model


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    model = build_clinical_encoder()

    model.summary()