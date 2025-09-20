import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Config ---
MODEL_PATH = r"D:\Projects\skin-age-detection\models\age_best_model.h5"
IMG_PATH = r"D:\Projects\skin-age-detection\datasets\UTKFace_10k_balanced\90_1_2_20170111221639268.jpg.chip.jpg"
IMG_SIZE = (256, 256)

# Sanity check
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image not found at {IMG_PATH}")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load + preprocess image
img = cv2.imread(IMG_PATH)                      # OpenCV loads in BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert to RGB
img = cv2.resize(img, IMG_SIZE)                 # Resize to model input
arr = np.array(img, dtype=np.float32)
arr = preprocess_input(arr)                     # ResNet-style preprocessing
arr = np.expand_dims(arr, axis=0)               # Add batch dimension

# Predict
predicted_age = model.predict(arr, verbose=0)[0][0]
print(f"{os.path.basename(IMG_PATH)} -> Predicted age: {predicted_age:.1f} years")
