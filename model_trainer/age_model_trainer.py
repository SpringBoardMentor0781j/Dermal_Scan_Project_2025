# =================================================================================================
# AUTOMATED SCRIPT FOR TRAINING AN AGE REGRESSION MODEL IN GOOGLE COLAB
# This script uses the EfficientNetB0 architecture and a tf.data pipeline for high performance.
# It will prioritize finding the dataset on Google Drive, and if not found, will automatically
# download and extract it before starting the training.
# =================================================================================================

# --- 1. SETUP AND IMPORTS ---

# Each line of code below is followed by a comment that explains its purpose in detail.
# This ensures that every step of the process is clearly understood.

# 'os' is a module that provides a way of using operating system dependent functionality.
# We will use this module to check for file and directory existence and to manage file paths.
import os
# 'random' is a module used for generating random numbers and making random selections.
# We will use this to shuffle our dataset for unbiased training and validation.
import random
# 'zipfile' is a module that provides tools to create, read, write, append, and list a ZIP file.
# We will use this to extract the downloaded dataset.
import zipfile

# 'tensorflow' is the core deep learning library we will use to build, train, and evaluate our model.
import tensorflow as tf
# 'layers', 'models', 'optimizers' are specific components from Keras for building and training our neural network.
from tensorflow.keras import layers, models, optimizers
# 'EfficientNetB0' is a specific, powerful, and modern pre-trained model architecture we will use for transfer learning.
from tensorflow.keras.applications import EfficientNetB0
# 'requests' is a library for making HTTP requests, which we'll use to download the dataset if it's not on Google Drive.
import requests


# ===============================
# 2. GLOBAL PARAMETERS
# ===============================

# DATASET_URL: This is the direct download link for the original UTKFace dataset ZIP file, used as a fallback.
DATASET_URL = "https://susanqq.github.io/UTKFace/data/UTKFace.zip"
# ZIP_FILE_NAME: This defines the name of the file that will be saved after downloading from the URL.
ZIP_FILE_NAME = "UTKFace.zip"
# LOCAL_DATASET_PATH: This is the standardized path we expect for the image folder in the Colab environment.
LOCAL_DATASET_PATH = "/content/UTKFace_resized"
# IMG_SIZE: Defines the width and height to which all images will be resized. 224x224 is the standard for EfficientNetB0.
IMG_SIZE = (224, 224)
# BATCH_SIZE: The number of images to process in each step (or batch) of training.
BATCH_SIZE = 16
# TRAIN_VAL_SPLIT: The proportion of the dataset to be used for training (80%). The rest (20%) will be for validation.
TRAIN_VAL_SPLIT = 0.8
# EPOCHS_HEAD: The number of training epochs for the first stage, where we only train the new top layers of the model.
EPOCHS_HEAD = 40
# EPOCHS_FINE: The number of training epochs for the second stage, where we fine-tune the entire model with a lower learning rate.
EPOCHS_FINE = 30


# ===============================
# 3. HELPER FUNCTIONS
# ===============================

def get_age_from_filename(fname):
    """
    Parses a filename to extract the age.
    The UTKFace dataset encodes metadata in the filename, e.g., 'age_gender_race_datetime.jpg'.

    Parameters:
        fname (str): The filename of the image (e.g., '25_0_0_20170116174525125.jpg.chip.jpg').

    Returns:
        int: The age extracted from the filename (e.g., 25).
    """
    # The filename is split by the '_' character. The first part is the age, which is converted to an integer.
    return int(fname.split('_')[0])

def prepare_tf_dataset(local_path):
    """
    Scans a directory for images, extracts labels from filenames, and splits them into training and validation sets.

    Parameters:
        local_path (str): The path to the directory containing all the image files.

    Returns:
        tuple: A tuple containing two lists: (training_files, validation_files). Each list contains (filepath, age) tuples.
    """
    # Initialize an empty list to store tuples of (filepath, age) for every valid image.
    all_files = []
    print(f"Scanning for images in '{local_path}'...")
    # 'os.walk' generates the file names in a directory tree by walking through it.
    for root, _, files in os.walk(local_path):
        # We loop through every file found in the directory.
        for fname in files:
            # We check if the file has a common image extension to ensure it's an image.
            if fname.lower().endswith(('.jpg','.jpeg','.png')):
                # We construct the full, absolute path to the image file.
                fpath = os.path.join(root, fname)
                try:
                    # We try to extract the age from the filename using our helper function.
                    age = get_age_from_filename(fname)
                    # If successful, we append the (filepath, age) tuple to our list.
                    all_files.append((fpath, age))
                except:
                    # If the filename is not in the expected format (e.g., 'age_...'), we skip it and continue.
                    continue
    
    # If no image files were found after scanning, we raise an error to stop the script.
    if not all_files:
        raise RuntimeError(f"FATAL: No valid image files were found in {local_path}")
    
    print(f"Found {len(all_files)} images.")
    # 'random.shuffle' shuffles the list of files in place. This ensures that when we split the data, both training
    # and validation sets will have a random, representative mix of all ages.
    random.shuffle(all_files)
    # We calculate the index at which to split the data into training and validation sets based on our specified ratio.
    split_idx = int(len(all_files) * TRAIN_VAL_SPLIT)
    # We return the two portions of the list: the training set and the validation set.
    return all_files[:split_idx], all_files[split_idx:]

def preprocess_image(path, label, augment=False):
    """
    Reads, decodes, resizes, and normalizes an image file. Optionally applies data augmentation.

    Parameters:
        path (str): The file path of the image.
        label (int): The age associated with the image.
        augment (bool): A boolean flag to indicate whether to apply random transformations to the image.

    Returns:
        tuple: A tuple containing the processed image tensor and its label tensor, ready for the model.
    """
    # 'tf.io.read_file' reads the raw binary contents of the image file.
    img = tf.io.read_file(path)
    # 'tf.image.decode_jpeg' decodes the raw data into a TensorFlow image tensor with 3 color channels (RGB).
    img = tf.image.decode_jpeg(img, channels=3)
    # 'tf.image.resize' resizes the image tensor to the target dimensions defined in our global parameters.
    img = tf.image.resize(img, IMG_SIZE)
    # 'tf.cast' changes the data type of the tensor to float32, and we normalize pixel values from the [0, 255] range to the [0, 1] range.
    img = tf.cast(img, tf.float32) / 255.0
    
    # If data augmentation is enabled for this dataset (typically for the training set), we apply random transformations.
    if augment:
        # 'tf.image.random_flip_left_right' randomly flips the image horizontally, which is a common augmentation technique.
        img = tf.image.random_flip_left_right(img)
        # 'tf.image.random_brightness' randomly adjusts the brightness of the image within a given range.
        img = tf.image.random_brightness(img, 0.2)
        # 'tf.image.random_contrast' randomly adjusts the contrast of the image.
        img = tf.image.random_contrast(img, 0.8, 1.2)
        
    # We return the processed image and its label, which is also cast to float32 for consistency.
    return img, tf.cast(label, tf.float32)

def build_dataset(file_list, augment=False):
    """
    Builds a complete, efficient tf.data.Dataset pipeline from a list of files and labels.

    Parameters:
        file_list (list): A list of (filepath, label) tuples.
        augment (bool): A boolean flag to enable or disable data augmentation for this dataset.

    Returns:
        tf.data.Dataset: A configured TensorFlow Dataset object that is shuffled, batched, and prefetched for optimal performance.
    """
    # 'zip(*file_list)' is a Python trick to unzip our list of tuples into two separate tuples: one for all paths, one for all labels.
    paths, labels = zip(*file_list)
    # 'tf.data.Dataset.from_tensor_slices' creates a dataset where each element is a (path, label) pair.
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    # 'ds.map' applies our 'preprocess_image' function to each element (each image) in the dataset.
    # 'num_parallel_calls=tf.data.AUTOTUNE' allows TensorFlow to use multiple CPU cores to process images in parallel, speeding things up.
    ds = ds.map(lambda x,y: preprocess_image(x,y,augment=augment),
                  num_parallel_calls=tf.data.AUTOTUNE)
    # 'ds.shuffle(1000)' shuffles the data to ensure the model sees images in a random order in each epoch, which improves training.
    # 'ds.batch(BATCH_SIZE)' groups the individual images and labels into batches of the size we specified.
    # 'ds.prefetch(tf.data.AUTOTUNE)' is a key performance optimization. It prepares the next batches of data on the CPU while the GPU is busy with the current batch.
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ===============================
# 4. MODEL DEFINITION
# ===============================

def build_regression_model():
    """
    Builds a Keras regression model using EfficientNetB0 as a pre-trained base (transfer learning).

    Returns:
        tf.keras.models.Model: The constructed Keras model, ready to be compiled.
    """
    # We instantiate the 'EfficientNetB0' model. 'weights='imagenet'' downloads weights pre-trained on the massive ImageNet dataset.
    # 'include_top=False' removes the original final classification layer, so we can add our own.
    # 'input_shape' specifies the dimensions of our input images.
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE,3))
    # 'base.trainable = False' freezes the weights of all the pre-trained layers. This is crucial for the first stage of training.
    base.trainable = False
    
    # We now add our custom layers on top of the EfficientNetB0 base. This part is our "ResNet-style head".
    # 'layers.GlobalAveragePooling2D' takes the output of the base model and averages the spatial features, creating a feature vector.
    x = layers.GlobalAveragePooling2D()(base.output)
    # 'layers.Dropout(0.3)' is a regularization technique that randomly deactivates 30% of neurons during training to prevent the model from becoming too specialized (overfitting).
    x = layers.Dropout(0.3)(x)
    # The final 'Dense' layer is our output layer. It has 1 neuron because we are predicting a single continuous value (the age).
    # It uses a 'linear' activation function, which is the standard for regression tasks as it can output any real number.
    out = layers.Dense(1, activation='linear')(x)
    
    # We create the final model by defining its input (from the base model) and its output (our custom final layer).
    model = models.Model(base.input, out)
    return model


# ===============================
# 5. TRAINING PIPELINE
# ===============================

def train_pipeline(train_ds, val_ds):
    """
    Executes the full two-stage training process: first training only the new layers (the "head"),
    and then fine-tuning the entire model with a lower learning rate.

    Parameters:
        train_ds (tf.data.Dataset): The fully configured training dataset.
        val_ds (tf.data.Dataset): The fully configured validation dataset.

    Returns:
        tf.keras.models.Model: The fully trained and saved Keras model.
    """
    # First, we build the model with the frozen backbone.
    model = build_regression_model()
    # We compile the model, which configures it for training.
    # We use the Adam optimizer with a relatively high learning rate (1e-3) for the initial training of the new layers.
    # 'loss='mse'' (Mean Squared Error) is a common loss function for regression tasks, penalizing larger errors more heavily.
    # 'metrics=['mae']' (Mean Absolute Error) will be monitored during training as it's more interpretable (average years of error).
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    
    print("\n--- Stage 1: Training head only ---")
    # 'model.fit' starts the training process for the specified number of epochs. We pass it the training and validation datasets.
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # Now, we unfreeze the entire model (including the EfficientNet base) to allow all layers to be trained.
    model.trainable = True
    # We must re-compile the model after changing the `trainable` status.
    # We use a much lower learning rate (1e-5) for fine-tuning. This is critical to avoid destroying the learned features from ImageNet.
    model.compile(optimizer=optimizers.Adam(1e-5), loss='mse', metrics=['mae'])
    
    print("\n--- Stage 2: Fine-tuning entire backbone ---")
    # We continue training the model, now fine-tuning all layers, for the specified number of fine-tuning epochs.
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

    # 'model.save' saves the final trained model to a single .h5 file, including its architecture, weights, and optimizer state.
    model.save("age_efficientnet_regression.h5")
    print("\nModel saved successfully as age_efficientnet_regression.h5")
    return model


# ===============================
# 6. EVALUATION FUNCTION
# ===============================

def evaluate_model(model, file_list, num_samples=200):
    """
    Evaluates the final model's performance on a random sample of the validation set and prints key metrics.

    Parameters:
        model (tf.keras.models.Model): The trained model to evaluate.
        file_list (list): The list of validation files, from which a random sample will be drawn.
        num_samples (int): The number of random images to test the model on.
    """
    print("\n--- Evaluating model performance ---")
    # 'random.sample' selects a random subset of files from the validation list for a final, quick evaluation.
    sample_files = random.sample(file_list, min(num_samples,len(file_list)))
    # 'zip(*sample_files)' separates the paths and labels from the list of tuples.
    paths, labels = zip(*sample_files)
    # We build a small, temporary dataset from the sample files for prediction (no data augmentation is used for evaluation).
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(lambda x,y: preprocess_image(x,y,augment=False))
    ds = ds.batch(BATCH_SIZE)
    
    # 'model.predict' runs the model on the evaluation dataset and returns the age predictions.
    preds = model.predict(ds)
    # 'tf.convert_to_tensor' converts the Python list of true labels to a TensorFlow tensor for calculations.
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    
    # We calculate the Mean Absolute Error (MAE) between the predictions and the true labels.
    mae = tf.reduce_mean(tf.abs(preds.flatten() - labels))
    # We calculate the percentage of predictions that are within 5 years of the true age, a useful accuracy metric.
    within_5 = tf.reduce_mean(tf.cast(tf.abs(preds.flatten() - labels) <= 5, tf.float32))
    
    print(f"\nEvaluation on {len(sample_files)} random validation images:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} years")
    print(f"Accuracy (within 5 years): {within_5*100:.2f}%")


# ===============================
# 7. MAIN EXECUTION BLOCK
# ===============================

# --- Step A: Ensure dataset is available in the correct location ---

# This 'if' statement checks if our target dataset directory does not already exist in the Colab environment.
if not os.path.isdir(LOCAL_DATASET_PATH):
    # This block of code runs only if the dataset is not found locally.
    print(f"Dataset folder '{LOCAL_DATASET_PATH}' not found in the local environment.")
    
    # Import necessary libraries for Drive and file copying, only when needed.
    from google.colab import drive
    import shutil
    
    # Mount Google Drive to check for the file there first.
    print("Mounting Google Drive to check for dataset...")
    drive.mount('/content/drive')
    
    # Define the exact path where the script will look for your ZIP file in Google Drive.
    DRIVE_ZIP_PATH = '/content/drive/MyDrive/UTKFace_resized.zip'
    
    # This 'if' statement checks if the preferred zip file exists at the specified path on your Google Drive.
    if os.path.exists(DRIVE_ZIP_PATH):
        # This block runs if the file is found on your Drive.
        print(f"Found '{os.path.basename(DRIVE_ZIP_PATH)}' in Google Drive.")
        print("Copying to Colab environment (this is much faster than uploading)...")
        # 'shutil.copy' performs a high-speed copy from your Drive to the local Colab filesystem.
        shutil.copy(DRIVE_ZIP_PATH, 'dataset.zip')
        print("Copy complete.")
        
        # Extract the contents of the copied zip file.
        print("Extracting dataset...")
        with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
            # All contents are extracted into the '/content/' directory.
            zip_ref.extractall('/content/')
        print("Extraction complete.")

    else:
        # This block runs if the file was NOT found on your Drive.
        print(f"Dataset not found in Google Drive at '{DRIVE_ZIP_PATH}'.")
        print(f"Falling back to downloading from original URL: {DATASET_URL}")
        try:
            # We use the 'requests' library to download the file from the web.
            with requests.get(DATASET_URL, stream=True) as r:
                r.raise_for_status() # This will raise an error if the download fails.
                with open(ZIP_FILE_NAME, 'wb') as f:
                    # We write the downloaded content to a local file in chunks to manage memory.
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded: {ZIP_FILE_NAME}")
            
            # We extract the newly downloaded file.
            print(f"Extracting '{ZIP_FILE_NAME}'...")
            with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall('/content/')
            
            # The original ZIP from the URL extracts to a folder named 'UTKFace'.
            # We rename it to 'UTKFace_resized' to match our standardized LOCAL_DATASET_PATH.
            os.rename('/content/UTKFace', LOCAL_DATASET_PATH)
            print(f"Renamed extracted folder to '{LOCAL_DATASET_PATH}'.")

        except Exception as e:
            # This catches any errors during the download or extraction process and stops the script.
            raise SystemExit(f"An error occurred during download/extraction: {e}")

    # This is a final check to ensure the dataset directory now exists before proceeding to training.
    if not os.path.isdir(LOCAL_DATASET_PATH):
        print(f"FATAL ERROR: The dataset folder '{LOCAL_DATASET_PATH}' could not be created.")
        raise SystemExit("Please check your ZIP file name on Drive or the download URL.")

else:
    # This 'else' block runs if the dataset directory was found at the very beginning.
    print(f"Dataset directory '{LOCAL_DATASET_PATH}' already exists. Skipping acquisition and extraction.")


# --- Step B: Prepare datasets, train model, and evaluate ---

# We call our helper function to get the lists of training and validation files from the dataset path.
train_files, val_files = prepare_tf_dataset(LOCAL_DATASET_PATH)
# We build the efficient tf.data.Dataset objects from these file lists. Augmentation is enabled for the training set only.
train_ds = build_dataset(train_files, augment=True)
val_ds = build_dataset(val_files, augment=False)

# We start the full, two-stage training pipeline. The fully trained model is returned and stored in the 'model' variable.
model = train_pipeline(train_ds, val_ds)
# Finally, we call our evaluation function to get a performance report on the newly trained model.
evaluate_model(model, val_files)

