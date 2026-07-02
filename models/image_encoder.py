import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# =====================================================
# IMAGE ENCODER
# =====================================================

IMAGE_SIZE = (224, 224, 3)


def build_image_encoder():

    # Pre-trained EfficientNetB0 backbone
    backbone = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SIZE
    )

    # Freeze backbone initially
    backbone.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE)

    x = backbone(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        128,
        activation="relu",
        name="image_features"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="Image_Encoder"
    )

    return model


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    model = build_image_encoder()

    model.summary()