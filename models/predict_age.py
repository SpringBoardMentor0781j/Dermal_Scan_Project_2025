# =================================================================================================
# SCRIPT TO PREDICT AGE FROM IMAGES USING A TRAINED KERAS MODEL
# This script provides functions to load a model, predict age for a single image, and
# evaluate the model's performance on a folder of images. It is designed to run on your local computer.
# =================================================================================================

# --- 0. IMPORT NECESSARY LIBRARIES ---

# 'os' is a module that provides functions for interacting with the operating system.
# We use it here to check if file and directory paths are valid.
import os
# 'numpy' is a fundamental package for numerical operations in Python.
# We use it for shuffling the list of images and for numerical calculations.
import numpy as np
# 'tensorflow' is the core deep learning library we will use.
import tensorflow as tf
# 'load_model' is the specific function from Keras (part of TensorFlow) to load a saved model file.
from tensorflow.keras.models import load_model

# --- 1. MODEL LOADER FUNCTION ---

def load_age_model(model_path):
    """
    Loads and returns the trained age prediction model from a .h5 file.

    This function is responsible for loading the saved Keras model. It includes a critical fix
    ('compile=False') to prevent errors caused by version mismatches between the training
    and prediction environments.

    Parameters:
        model_path (str): This is the full path to the saved Keras .h5 model file.

    Returns:
        tf.keras.Model: The loaded Keras model, ready for making predictions.
    """
    # This line checks if the file at the specified path actually exists.
    if not os.path.exists(model_path):
        # If the file is not found, a FileNotFoundError is raised with a helpful message, and the script stops.
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # This print statement provides feedback to the user that the loading process is starting.
    print(f"Loading model from: {model_path}")
    # This is the core line for loading the model.
    # 'compile=False' is the crucial fix: it tells Keras to only load the model's architecture
    # and its learned weights, but to ignore the optimizer state and loss function configuration.
    # This prevents the deserialization error you were seeing.
    model = load_model(model_path, compile=False)
    # This print statement confirms that the model was loaded successfully.
    print("Model loaded successfully.")
    # The function returns the loaded model object.
    return model

# --- 2. SINGLE IMAGE PREDICTION FUNCTION ---

def predict_age(model, image_path):
    """
    Predicts the age from a single image file, assuming the image is already sized correctly (e.g., 224x224).

    This function handles the entire prediction pipeline for one image: reading, preprocessing,
    predicting, and post-processing the result.

    Parameters:
        model (tf.keras.Model): The pre-loaded Keras model object returned by `load_age_model`.
        image_path (str): The full path to the image file for which you want to predict the age.

    Returns:
        float or None: The predicted age as a floating-point number, or None if the image file is not found.
    """
    # This line checks if the image file exists at the given path.
    if not os.path.exists(image_path):
        # If the file is missing, an error is printed, and the function returns None.
        print(f"Error: Image file not found at {image_path}")
        return None

    # 'tf.io.read_file' reads the entire content of the file into a raw bytes tensor.
    img = tf.io.read_file(image_path)
    # 'tf.image.decode_jpeg' decodes the raw image data into a proper image tensor with 3 color channels (RGB).
    img = tf.image.decode_jpeg(img, channels=3)
    
    # NOTE: This function assumes the image is already correctly sized (e.g., 224x224), so a resizing step is skipped.
    
    # 'tf.cast' changes the data type of the image tensor to float32, which is required for calculations.
    # We then divide by 255.0 to normalize the pixel values from the [0, 255] range to the [0, 1] range.
    img = tf.cast(img, tf.float32) / 255.0
    # The model expects a "batch" of images as input, even for a single prediction.
    # 'tf.expand_dims' adds a new dimension at the beginning (axis=0), changing the shape from (H, W, C) to (1, H, W, C).
    img_batch = tf.expand_dims(img, axis=0)
    
    # 'model.predict' runs the model on the input image batch and returns the prediction(s). 'verbose=0' keeps the output clean.
    prediction = model.predict(img_batch, verbose=0)
    # The prediction is often returned in a nested array (e.g., [[25.5]]). 'flatten()[0]' extracts the single numerical value.
    predicted_age = prediction.flatten()[0]
    
    # This is a safety check to ensure the model doesn't output a nonsensical negative age.
    # 'max(0, ...)' will return 0 if the prediction is negative, otherwise it returns the prediction itself.
    predicted_age = max(0, predicted_age)
    
    # The final, processed prediction is returned by the function.
    return predicted_age

# --- 3. BATCH FOLDER EVALUATION FUNCTION ---

def predict_folder(model, folder_path, num_images=200, batch_size=64):
    """
    Predicts ages for a random batch of images from a folder and calculates the Mean Absolute Error (MAE).

    This function is for evaluating the model's performance. It efficiently processes multiple images
    using a tf.data.Dataset pipeline and compares the predictions to the true ages parsed from the filenames.

    Parameters:
        model (tf.keras.Model): The pre-loaded Keras model.
        folder_path (str): The path to the directory containing the test images.
        num_images (int): The number of random images to select from the folder for the test.
        batch_size (int): The number of images to process at once during prediction, for efficiency.
    """
    # This line checks if the provided path is a valid directory.
    if not os.path.isdir(folder_path):
        # If not, an error is raised, and the script stops.
        raise NotADirectoryError(f"Image directory not found: {folder_path}")

    # 'os.listdir' gets a list of all filenames in the directory.
    all_images = os.listdir(folder_path)
    # 'np.random.shuffle' randomizes the order of the list in place.
    np.random.shuffle(all_images)
    # This line selects a subset of the shuffled images, ensuring we don't try to select more images than exist.
    selected = all_images[:min(num_images, len(all_images))]
    # This line creates a list of full, absolute paths for each selected image filename.
    selected_paths = [os.path.join(folder_path, f) for f in selected]

    # This is a small helper function defined inside `predict_folder` to keep the code organized.
    def load_and_preprocess(path):
        # It reads the raw image file.
        img = tf.io.read_file(path)
        # It decodes the image into a 3-channel tensor.
        img = tf.image.decode_jpeg(img, channels=3)
        # It normalizes the pixel values to the [0, 1] range.
        img = tf.cast(img, tf.float32) / 255.0
        # It returns the preprocessed image tensor.
        return img

    # 'tf.data.Dataset.from_tensor_slices' creates a TensorFlow Dataset from the list of file paths.
    ds = tf.data.Dataset.from_tensor_slices(selected_paths)
    # '.map' applies our 'load_and_preprocess' function to each file path in the dataset.
    # 'num_parallel_calls' allows TensorFlow to process multiple images in parallel for better performance.
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # '.batch' groups the images into batches of the specified size.
    ds = ds.batch(batch_size)
    # '.prefetch' is a performance optimization that prepares the next batch of data while the current one is being processed.
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # 'model.predict' runs the model on the entire dataset and returns all predictions as a single NumPy array.
    preds = model.predict(ds, verbose=0).flatten()
    # 'np.maximum' is a fast way to apply the "no negative age" rule to the entire array of predictions at once.
    preds = np.maximum(preds, 0)

    # We initialize an empty list to store the absolute error for each prediction.
    errors = []
    # We loop through both the list of image paths and the corresponding predictions at the same time.
    for path, pred in zip(selected_paths, preds):
        # 'os.path.basename' gets just the filename from the full path.
        fname = os.path.basename(path)
        # This 'try...except' block handles cases where a filename might not be in the correct format.
        try:
            # The actual age is parsed from the first part of the filename (e.g., '35_...').
            actual = int(fname.split("_")[0])
            # The absolute error is the difference between the prediction and the actual age.
            error = abs(pred - actual)
            # We add this error to our list.
            errors.append(error)
            # A print statement provides real-time feedback on each image.
            print(f"{fname}: actual={actual}, predicted={pred:.1f}, error={error:.1f}")
        except Exception:
            # If the age can't be parsed, a message is printed, and the file is skipped.
            print(f"{fname}: Could not parse actual age from filename.")

    # This condition checks if we were able to make any valid predictions.
    if errors:
        # If so, a final summary report is printed.
        print("\n---------------------------------")
        print(f"Test complete on {len(errors)} images.")
        # 'np.mean(errors)' calculates the Mean Absolute Error from our list of individual errors.
        print(f"Mean Absolute Error: {np.mean(errors):.2f} years")
        print("---------------------------------")
    else:
        # If no valid predictions could be compared, this message is shown.
        print("No valid predictions were made.")

# --- 4. EXAMPLE USAGE BLOCK ---

# This 'if' block ensures that the code inside it only runs when the script is executed directly
# (e.g., by running 'python predict_age.py' in the terminal).
if __name__ == "__main__":
    # This line calls our model loading function with the path to your saved model file.
    # The 'r' before the string makes it a "raw" string, which helps avoid issues with backslashes in Windows file paths.
    model = load_age_model(r"D:\Projects\skin-age-detection\models\age_mobilenet_regression_stratified.h5")

    # --- Test on a single image ---
    # This line defines the path to a single image you want to test.
    single_image_test_path = r"C:\Users\Hi\Pictures\Camera Roll\WIN_20250920_20_30_10_Pro.jpg"
    # A print statement to clearly mark this section of the output.
    print("\n--- Testing on a single image ---")
    # This line calls our single image prediction function.
    predicted_age = predict_age(model, single_image_test_path)
    # This condition checks if the prediction was successful (i.e., not None).
    if predicted_age is not None:
        # If successful, the predicted age is printed, formatted to one decimal place.
        print(f"Predicted age for single image is: {predicted_age:.1f} years\n")

    # --- Run batch evaluation on a folder ---
    # A print statement to mark this section of the output.
    print("--- Running batch evaluation on folder ---")
    # This line calls our batch evaluation function, pointing it to the directory containing your test images.
    predict_folder(model, r"D:\Projects\skin-age-detection\datasets\UTKFace_resized")

