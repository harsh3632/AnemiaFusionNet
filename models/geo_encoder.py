import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# GEO ENCODER
# =====================================================

INPUT_FEATURES = 1

def build_geo_encoder():

    inputs = tf.keras.Input(
        shape=(INPUT_FEATURES,),
        name="geo_input"
    )

    x = layers.Dense(
        32,
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.30)(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="geo_features"
    )(x)

    model = models.Model(
        inputs,
        outputs=x,
        name="Geo_Encoder"
    )

    return model


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    model = build_geo_encoder()

    model.summary()