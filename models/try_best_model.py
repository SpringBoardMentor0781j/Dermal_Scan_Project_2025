import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the model
model = tf.keras.models.load_model('best_model.h5')

# Preprocess image
img = Image.open(r'D:\Projects\skin-age-detection\datasets\UTKFace_resized\40_0_1_20170113184933016.jpg.chip.jpg').convert('RGB').resize((256, 256))
arr = np.array(img, dtype=np.float32)
arr = preprocess_input(arr)
arr = np.expand_dims(arr, 0)

# Predict age
predicted_age = model.predict(arr)[0][0]
print(f"Predicted age: {predicted_age:.1f} years")
