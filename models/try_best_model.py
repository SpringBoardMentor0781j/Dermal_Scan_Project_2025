import os, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# --- Parameters ---
DESIRED_CLASSES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
IMG_SIZE = (224,224)

def build_model(num_classes=4):
    """Rebuild architecture to match training before loading weights."""
    backbone = EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=backbone.input, outputs=preds)

def load_trained_model(model_path="efficientnet_b0_face.h5"):
    """Rebuild architecture and load weights only."""
    model = build_model(len(DESIRED_CLASSES))
    model.load_weights(model_path)
    return model

def predict_image(model, image_path):
    """Predict class for a single image using loaded model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    return DESIRED_CLASSES[pred_idx], confidence

def predict_folder(folder_path, model_path="efficientnet_b0_face.h5"):
    """Predict all images in a folder."""
    model = load_trained_model(model_path)

    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
    if not files:
        raise RuntimeError(f"No images found in {folder_path}")

    for f in files:
        img_path = os.path.join(folder_path, f)
        label, conf = predict_image(model, img_path)
        print(f"{f}: {label} ({conf:.2f})")

# --- Example usage ---
if __name__ == "__main__":
    remote_dir = r"D:\Projects\skin-age-detection\datasets\UTKFace_resized"
    predict_folder(remote_dir, "D:\Projects\skin-age-detection\models\efficientnet_b0_face.h5")
