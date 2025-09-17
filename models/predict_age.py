# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. Model Loader ---
def load_age_model(model_path="age_predictor_utkface.h5"):
    """
    Load and return the trained age prediction model.
    Args:
        model_path: path to the saved Keras .h5 model
    Returns:
        Keras model
    """
    # Check if the model file exists before trying to load it.
    if not os.path.exists(model_path):
        # Raise an error if the file is not found.
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from: {model_path}")
    # Load the pre-trained Keras model from the specified file path.
    model = load_model(model_path)
    print("Model loaded successfully.")
    # Return the loaded model object.
    return model

# --- 2. Stable Single Image Prediction Function ---
def predict_age(model, image_path):
    """
    Predicts the age from a single, pre-sized (224x224) image file.
    Args:
        model (tf.keras.Model): The pre-loaded Keras model.
        image_path (str): The file path to the image.
    Returns:
        float or None: The predicted age, or None if the file doesn't exist.
    """
    # Check if the image file exists at the given path.
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    # Use TensorFlow's I/O functions to read the image file.
    img = tf.io.read_file(image_path)
    # Decode the image file into a tensor with 3 color channels (RGB).
    img = tf.image.decode_jpeg(img, channels=3)
    
    # The resizing step is now removed, assuming input is already 224x224.
    
    # Normalize pixel values from the [0, 255] range to the [0.0, 1.0] range.
    img = tf.cast(img, tf.float32) / 255.0
    # Add a batch dimension to the tensor; model expects input shape (1, 224, 224, 3).
    img_batch = tf.expand_dims(img, axis=0)
    
    # Use the model to predict the age from the preprocessed image batch.
    prediction = model.predict(img_batch, verbose=0)
    # The prediction is a nested array (e.g., [[35.4]]), so extract the single value.
    predicted_age = prediction.flatten()[0]
    
    # Ensure the predicted age is not negative by clamping it at 0.
    predicted_age = max(0, predicted_age)
    
    # Return the final predicted age.
    return predicted_age

# --- 3. Batch Folder Evaluation Function ---
def predict_folder(model, folder_path, num_images=200, batch_size=64):
    """
    Predict ages for a batch of images in a folder and calculate the Mean Absolute Error.
    Args:
        model: pre-loaded Keras model
        folder_path: path to folder containing images
        num_images: number of random images to test
        batch_size: batch size for prediction
    """
    # Check if the provided folder path is a valid directory.
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Image directory not found: {folder_path}")

    # Get a list of all files in the directory.
    all_images = os.listdir(folder_path)
    # Shuffle the list of images to get a random sample.
    np.random.shuffle(all_images)
    # Select a subset of images to test.
    selected = all_images[:min(num_images, len(all_images))]
    # Create the full file paths for the selected images.
    selected_paths = [os.path.join(folder_path, f) for f in selected]

    # Define a helper function to load and preprocess images for the dataset pipeline.
    def load_and_preprocess(path):
        img = tf.io.read_file(path) # Read the image file.
        img = tf.image.decode_jpeg(img, channels=3) # Decode as a 3-channel image.
        
        # Resizing step is removed here as well.
        
        img = tf.cast(img, tf.float32) / 255.0 # Normalize pixel values.
        return img

    # Create a TensorFlow Dataset from the list of file paths for efficient processing.
    ds = tf.data.Dataset.from_tensor_slices(selected_paths)
    # Apply the preprocessing function to each image in parallel.
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # Group the images into batches.
    ds = ds.batch(batch_size)
    # Prefetch data to improve performance by overlapping data preparation and model execution.
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Get predictions for the entire dataset.
    preds = model.predict(ds, verbose=0).flatten()
    # Clamp any negative predictions to 0.
    preds = np.maximum(preds, 0)

    # Calculate the error for each prediction.
    errors = []
    for path, pred in zip(selected_paths, preds):
        fname = os.path.basename(path) # Get the filename from the path.
        try:
            # The actual age is encoded in the filename (e.g., "35_...jpg").
            actual = int(fname.split("_")[0])
            error = abs(pred - actual) # Calculate the absolute error.
            errors.append(error)
            print(f"{fname}: Actual={actual}, Predicted={pred:.1f}, Error={error:.1f}")
        except Exception:
            print(f"{fname}: Could not parse actual age from filename.")

    # Print the final evaluation summary.
    if errors:
        print("\n---------------------------------")
        print(f"Test complete on {len(errors)} images")
        print(f"Mean Absolute Error: {np.mean(errors):.2f} years") # Calculate and print the average error.
        print("---------------------------------")
    else:
        print("No valid predictions were made.")

# --- Example Usage ---
if __name__ == "__main__":
    # Load the model from the .h5 file.
    model = load_age_model("age_predictor_utkface.h5")

    # --- DEMONSTRATE THE STABLE `predict_age` FUNCTION ---
    # IMPORTANT: Change this to the path of a single image for testing.
    single_image_test_path = r"D:\Projects\skin-age-detection\datasets\UTKFace_resized\88_0_0_20170111210626829.jpg.chip.jpg"
    print("\n--- Testing on a single image ---")
    predicted_age = predict_age(model, single_image_test_path) # Call the single prediction function.
    if predicted_age is not None:
        print(f"✅ The predicted age for the single image is: {predicted_age:.1f} years\n")
r'''
    # --- DEMONSTRATE THE `predict_folder` BATCH EVALUATION ---
    # This runs the batch prediction and calculates the error.
    print("--- Running batch evaluation on folder ---")
    predict_folder(model, r"D:\Projects\skin-age-detection\datasets\UTKFace_resized")

    '''