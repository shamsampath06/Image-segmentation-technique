"""
unet.py
-------
U-Net Semantic Segmentation Architecture
=========================================
Implements the classic U-Net encoder-decoder architecture with skip connections.

Architecture overview:
  Encoder (Contracting Path):
    Block 1 : Conv(64)  → Conv(64)  → MaxPool
    Block 2 : Conv(128) → Conv(128) → MaxPool
    Block 3 : Conv(256) → Conv(256) → MaxPool
    Block 4 : Conv(512) → Conv(512) → MaxPool

  Bottleneck:
    Conv(1024) → Conv(1024)

  Decoder (Expanding Path):
    Block 4 : UpConv(512) → Concat skip4 → Conv(512)  → Conv(512)
    Block 3 : UpConv(256) → Concat skip3 → Conv(256)  → Conv(256)
    Block 2 : UpConv(128) → Concat skip2 → Conv(128)  → Conv(128)
    Block 1 : UpConv(64)  → Concat skip1 → Conv(64)   → Conv(64)

  Output:
    Conv(1, kernel=1, activation='sigmoid')   [binary segmentation]
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ─── Loss Functions ────────────────────────────────────────────────────────────

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Soft Dice Loss — well-suited for binary segmentation with class imbalance.
    dice_loss = 1 - (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)
    """
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combined Binary Cross-Entropy + Dice Loss for robust training."""
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# ─── Building Blocks ──────────────────────────────────────────────────────────

def conv_block(x: tf.Tensor, filters: int, name_prefix: str = "") -> tf.Tensor:
    """Two consecutive Conv2D → BatchNorm → ReLU layers."""
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block(x: tf.Tensor, filters: int, name_prefix: str = ""):
    """
    Encoder block: conv_block → MaxPool
    Returns (skip_connection, pooled_output)
    """
    skip = conv_block(x, filters, name_prefix=name_prefix)
    pool = layers.MaxPooling2D((2, 2))(skip)
    return skip, pool


def decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    name_prefix: str = ""
) -> tf.Tensor:
    """
    Decoder block: UpSampling2D → Concat(skip) → conv_block
    """
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, name_prefix=name_prefix)
    return x


# ─── U-Net Model ──────────────────────────────────────────────────────────────

def build_unet(input_shape: tuple = (128, 128, 3), num_classes: int = 1) -> Model:
    """
    Build and compile the full U-Net model.

    Parameters
    ----------
    input_shape : tuple  — (H, W, C), default (128, 128, 3)
    num_classes : int    — 1 for binary, >1 for multi-class

    Returns
    -------
    model : tf.keras.Model  — compiled U-Net
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Encoder ───────────────────────────────────────────────────────────────
    s1, p1 = encoder_block(inputs, filters=64,   name_prefix="enc1")
    s2, p2 = encoder_block(p1,     filters=128,  name_prefix="enc2")
    s3, p3 = encoder_block(p2,     filters=256,  name_prefix="enc3")
    s4, p4 = encoder_block(p3,     filters=512,  name_prefix="enc4")

    # ── Bottleneck ────────────────────────────────────────────────────────────
    bottleneck = conv_block(p4, filters=1024, name_prefix="bottleneck")

    # ── Decoder ───────────────────────────────────────────────────────────────
    d4 = decoder_block(bottleneck, skip=s4, filters=512,  name_prefix="dec4")
    d3 = decoder_block(d4,         skip=s3, filters=256,  name_prefix="dec3")
    d2 = decoder_block(d3,         skip=s2, filters=128,  name_prefix="dec2")
    d1 = decoder_block(d2,         skip=s1, filters=64,   name_prefix="dec1")

    # ── Output ────────────────────────────────────────────────────────────────
    if num_classes == 1:
        # Binary segmentation → sigmoid
        outputs = layers.Conv2D(
            1, (1, 1), activation="sigmoid", name="output_mask"
        )(d1)
        loss_fn = bce_dice_loss
        metrics  = ["accuracy"]
    else:
        # Multi-class segmentation → softmax
        outputs = layers.Conv2D(
            num_classes, (1, 1), activation="softmax", name="output_mask"
        )(d1)
        loss_fn = "sparse_categorical_crossentropy"
        metrics  = ["accuracy"]

    model = Model(inputs, outputs, name="U-Net")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=metrics,
    )
    return model


def model_summary() -> None:
    """Print the U-Net model summary to console."""
    model = build_unet()
    model.summary()


if __name__ == "__main__":
    model_summary()
    print("\nTotal parameters:", model_summary())
