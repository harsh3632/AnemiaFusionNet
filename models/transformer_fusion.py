import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# TRANSFORMER BLOCK
# =====================================================

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout_rate=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):

        attention = self.att(inputs, inputs)

        attention = self.dropout1(
            attention,
            training=training
        )

        out1 = self.layernorm1(inputs + attention)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.layernorm2(out1 + ffn_output)

    def get_config(self):

        config = super().get_config()

        config.update({

            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate

        })

        return config


# =====================================================
# MULTIMODAL FUSION MODEL
# =====================================================

def build_transformer_fusion():

    image_input = tf.keras.Input(
        shape=(128,),
        name="image_features"
    )

    clinical_input = tf.keras.Input(
        shape=(128,),
        name="clinical_features"
    )

    geo_input = tf.keras.Input(
        shape=(128,),
        name="geo_features"
    )

    image = layers.Reshape((1, 128))(image_input)
    clinical = layers.Reshape((1, 128))(clinical_input)
    geo = layers.Reshape((1, 128))(geo_input)

    x = layers.Concatenate(axis=1)([
        image,
        clinical,
        geo
    ])

    x = TransformerBlock(
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout_rate=0.1
    )(x)

    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(
        128,
        activation="relu",
        name="fusion_features"
    )(x)

    return models.Model(
        inputs=[
            image_input,
            clinical_input,
            geo_input
        ],
        outputs=outputs,
        name="Transformer_Fusion"
    )


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    model = build_transformer_fusion()

    model.summary()