# Save this file as: models/predict_feature.py
# Backend module, no Streamlit dependency.

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model as keras_load_model

# --- Config ---
IMG_SIZE = (224, 224)
DESIRED_FEATURES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']

def build_model_architecture(num_classes: int):
    """Build model skeleton for weight-only loading."""
    backbone = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=backbone.input, outputs=preds)

def load_model(model_path: str):
    """
    Load model from .h5. If file contains only weights,
    architecture is rebuilt first.
    """
    try:
        # Attempt to load full model
        return keras_load_model(model_path, compile=False)
    except Exception:
        # Fall back to skeleton + weights
        model = build_model_architecture(num_classes=len(DESIRED_FEATURES))
        model.load_weights(model_path)
        return model

def prepare_image(img_array: np.ndarray):
    """Preprocess raw image for EfficientNet input."""
    if img_array.ndim == 2:  # grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    img_resized = cv2.resize(img_array, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_batch = np.expand_dims(img_resized, axis=0)
    return preprocess_input(img_batch)

def predict_features(model: tf.keras.Model, image_array: np.ndarray):
    """Predict facial features as probabilities."""
    prepared = prepare_image(image_array)
    probabilities = model(prepared, training=False).numpy()[0]
    return dict(zip(DESIRED_FEATURES, probabilities))
