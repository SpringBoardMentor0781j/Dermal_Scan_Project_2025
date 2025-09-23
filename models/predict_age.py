import os
import numpy as np
import tensorflow as tf
from model_loader import load_keras_model



def predict_age(model, image_path):
    """Predict the age from a single image file."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img_batch = tf.expand_dims(img, axis=0)

    prediction = model.predict(img_batch, verbose=0)
    predicted_age = max(0, prediction.flatten()[0])
    return predicted_age


def predict_folder(model, folder_path, num_images=200, batch_size=64):
    """Evaluate model on a random batch of images from a folder."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Image directory not found: {folder_path}")

    all_images = os.listdir(folder_path)
    np.random.shuffle(all_images)
    selected = all_images[:min(num_images, len(all_images))]
    selected_paths = [os.path.join(folder_path, f) for f in selected]

    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(selected_paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    preds = model.predict(ds, verbose=0).flatten()
    preds = np.maximum(preds, 0)

    errors = []
    for path, pred in zip(selected_paths, preds):
        fname = os.path.basename(path)
        try:
            actual = int(fname.split("_")[0])
            error = abs(pred - actual)
            errors.append(error)
            print(f"{fname}: actual={actual}, predicted={pred:.1f}, error={error:.1f}")
        except Exception:
            print(f"{fname}: Could not parse actual age from filename.")

    if errors:
        print("\n---------------------------------")
        print(f"Test complete on {len(errors)} images.")
        print(f"Mean Absolute Error: {np.mean(errors):.2f} years")
        print("---------------------------------")
    else:
        print("No valid predictions were made.")


if __name__ == "__main__":
    model = load_keras_model(
        r"D:\Projects\skin-age-detection\models\age_efficientnet_regression_stratified.h5"
    )

    test_image = r"C:\Users\Hi\Pictures\Camera Roll\WIN_20250920_20_30_10_Pro.jpg"
    print("\n--- Testing on a single image ---")
    age = predict_age(model, test_image)
    if age is not None:
        print(f"Predicted age: {age:.1f} years\n")

    print("--- Running batch evaluation on folder ---")
    predict_folder(model, r"D:\Projects\skin-age-detection\datasets\UTKFace_resized")
