# python
import os # Provides a way of using operating system dependent functionality.
import random # Implements pseudo-random number generators for various distributions.
import tensorflow as tf # Imports the main TensorFlow library for machine learning tasks.
import numpy as np # Imports NumPy for numerical operations, especially with arrays.
import matplotlib.pyplot as plt # Imports Matplotlib for creating static, animated, and interactive visualizations.
from tensorflow.keras.preprocessing.image import load_img, img_to_array # Imports specific functions for loading and converting images.

# --- Configuration ---
DESIRED_CLASSES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles'] # Defines the class labels your model predicts.
IMG_SIZE = (224, 224) # Sets the target image size for preprocessing to match the model's input.
MODEL_PATH = "D:\Projects\skin-age-detection\models\mobilenet_effnet_head.h5" # Defines the path to your saved Keras model file.
IMAGE_DIR = r"datasets/UTKFace_resized" # Sets the directory from which to pull random images for prediction.
NUM_IMAGES = 20 # Specifies the number of random images to select and predict on.

def load_model_safe(model_path):
    """
    Safely loads a Keras model with error handling for the file path.

    Parameters:
    model_path (str): The relative or absolute path to the .h5 model file.

    Returns:
    tensorflow.keras.Model: The loaded Keras model object.
    """
    if not os.path.exists(model_path): # Checks if the model file exists at the given path.
        raise FileNotFoundError(f"Model file not found at: {model_path}") # Raises an error if the file is not found.
    print(f"Loading model from: {model_path}") # Prints a confirmation message with the model path.
    return tf.keras.models.load_model(model_path) # Loads and returns the Keras model.

def predict_image_array(img_array, model):
    """
    Predicts class probabilities from a preprocessed image array.

    Parameters:
    img_array (np.ndarray): The preprocessed image data as a NumPy array.
    model (tensorflow.keras.Model): The loaded Keras model to use for prediction.

    Returns:
    tuple: A tuple containing the predicted label (str), its confidence (float), and all class probabilities (np.ndarray).
    """
    img_array_exp = np.expand_dims(img_array, axis=0) # Adds a batch dimension to the image array to match the model's expected input shape (1, height, width, channels).
    preds = model.predict(img_array_exp, verbose=0)[0] # Runs the prediction and gets the output for the first (and only) image in the batch.
    return preds # Returns the raw prediction array containing probabilities for each class.

def load_and_preprocess_image(image_path):
    """
    Loads an image file from a path and preprocesses it for the model.

    Parameters:
    image_path (str): The full path to the image file.

    Returns:
    tuple: A tuple containing the preprocessed image as a NumPy array and the original PIL image object.
    """
    img = load_img(image_path, target_size=IMG_SIZE) # Loads the image from the path and resizes it to the target size.
    img_array = img_to_array(img) / 255.0 # Converts the PIL image to a NumPy array and rescales pixel values to the [0, 1] range.
    return img_array, img # Returns both the array and the original image object.

if __name__ == "__main__":
    # --- Load Model ---
    model = load_model_safe(MODEL_PATH) # Loads the specified model using the safe loading function.

    # --- Select Random Images ---
    if not os.path.isdir(IMAGE_DIR): # Checks if the specified image directory exists.
        raise NotADirectoryError(f"Image directory not found: {IMAGE_DIR}") # Raises an error if the directory does not exist.
    
    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] # Lists all files in the directory and filters for common image extensions.
    random_images = random.sample(all_images, min(NUM_IMAGES, len(all_images))) # Randomly selects a sample of images, ensuring not to request more images than are available.

    # --- Create Visualization Grid ---
    # Calculates the number of rows and columns needed to display all images in a grid.
    cols = 5 # Sets a fixed number of columns for the grid.
    rows = (len(random_images) + cols - 1) // cols # Calculates the required number of rows.
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows)) # Creates a Matplotlib figure and a grid of subplots.
    axes = axes.flatten() # Flattens the 2D array of axes into a 1D array for easy iteration.

    # --- Process and Display Each Image ---
    for i, img_name in enumerate(random_images): # Iterates through the selected random image filenames.
        img_path = os.path.join(IMAGE_DIR, img_name) # Constructs the full path to the current image.
        try:
            img_array, img = load_and_preprocess_image(img_path) # Loads and preprocesses the image.
            predictions = predict_image_array(img_array, model) # Gets the prediction percentages for all classes.

            # --- Format the title with all class percentages ---
            title_str = "" # Initializes an empty string for the plot title.
            for j, class_name in enumerate(DESIRED_CLASSES): # Iterates through the class names and their corresponding prediction values.
                title_str += f"{class_name}: {predictions[j]*100:.1f}%\n" # Appends each class name and its formatted percentage to the title string.
            
            # --- Display the image and its prediction title ---
            ax = axes[i] # Selects the current subplot axis.
            ax.imshow(img) # Displays the image on the subplot.
            ax.set_title(title_str.strip(), fontsize=8) # Sets the formatted title for the subplot and removes trailing whitespace.
            ax.axis('off') # Hides the x and y axes for a cleaner look.

        except Exception as e: # Catches any errors that occur during image processing.
            print(f"Could not process {img_path}: {e}") # Prints an error message if an image fails to load or process.
            axes[i].axis('off') # Hides the axis for the failed image subplot.

    # --- Clean up and show the plot ---
    # Hides any unused subplots if the number of images is not a perfect multiple of the number of columns.
    for j in range(len(random_images), len(axes)):
        axes[j].axis('off') # Turns off the axis for the empty subplot.

    plt.tight_layout() # Adjusts subplot params so that subplots are nicely fit in the figure.
    plt.show() # Displays the entire figure with all the subplots.