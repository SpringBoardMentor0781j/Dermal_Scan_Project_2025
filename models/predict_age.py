# models/predict_age.py
import os
import numpy as np
import tensorflow as tf
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

def load_model(model_path=None):
    """Load a Keras model for inference (no compile)."""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "age_mobilenet_regression_stratified_old.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)

def preprocess_input(input_data):
    """
    Convert path/bytes/ndarray to normalized (1,H,W,3) tensor.
    """
    if isinstance(input_data, str):
        img = tf.io.read_file(input_data)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
    elif isinstance(input_data, bytes):
        arr = np.frombuffer(input_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 2:
            img = cv2.cvtColor(input_data, cv2.COLOR_GRAY2RGB)
        elif input_data.shape[2] == 4:
            img = cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGB)
        else:
            img = input_data
        img = img.astype(np.float32) / 255.0
    else:
        raise TypeError("Unsupported input type. Must be path, bytes, or np.ndarray.")

    if not isinstance(img, tf.Tensor):
        img = tf.convert_to_tensor(img, dtype=tf.float32)

    return tf.expand_dims(img, axis=0)

def predict_age(model, input_data):
    """Predict age from a single image."""
    img_tensor = preprocess_input(input_data)
    prediction = model(img_tensor, training=False).numpy().flatten()[0]
    return max(0, float(prediction))

def predict_folder(model, folder_path, num_images=200, batch_size=64):
    """Evaluate model on random sample from folder."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Image directory not found: {folder_path}")

    all_images = os.listdir(folder_path)
    np.random.shuffle(all_images)
    selected = all_images[:min(num_images, len(all_images))]
    selected_paths = [os.path.join(folder_path, f) for f in selected]

    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    ds = tf.data.Dataset.from_tensor_slices(selected_paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    preds = model.predict(ds, verbose=0).flatten()
    preds = np.maximum(preds, 0)

    # Try to parse actual ages from filenames
    actuals, valid_preds = [], []
    for path, pred in zip(selected_paths, preds):
        fname = os.path.basename(path)
        try:
            actual = int(fname.split("_")[0])
            actuals.append(actual)
            valid_preds.append(pred)
            print(f"{fname}: actual={actual}, predicted={pred:.1f}, error={abs(pred-actual):.1f}")
        except Exception:
            print(f"{fname}: Could not parse actual age.")

    if actuals:
        errors = np.abs(np.array(valid_preds) - np.array(actuals))
        print("\n---------------------------------")
        print(f"Test complete on {len(errors)} images.")
        print(f"Mean Absolute Error: {np.mean(errors):.2f} years")
        print("---------------------------------")
    else:
        print("No valid predictions were made.")

if __name__ == "__main__":
    model = load_model()
    # Replace with a clean test path
    test_image = os.path.join(DATASETS_DIR, "UTKFace_resized", "1_0_0_20161219140642920.jpg.chip.jpg")
    age = predict_age(model, test_image)
    print(f"Predicted age: {age:.1f}")
