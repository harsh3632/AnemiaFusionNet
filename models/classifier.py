import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# FINAL CLASSIFIER
# =====================================================

def build_classifier():

    fusion_input = tf.keras.Input(
        shape=(128,),
        name="fusion_features"
    )

    x = layers.Dense(
        64,
        activation="relu"
    )(fusion_input)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="anemia_prediction"
    )(x)

    model = models.Model(
        inputs=fusion_input,
        outputs=output,
        name="Anemia_Classifier"
    )

    return model


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    model = build_classifier()

    model.summary()