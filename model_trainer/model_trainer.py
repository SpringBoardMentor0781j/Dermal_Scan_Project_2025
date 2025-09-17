import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import matplotlib.pyplot as plt
import shutil


def create_onehot_facial_df(data_dir):
    """
    Creates a pandas DataFrame with image paths and one-hot encoded class labels,
    filtering for specific categories.

    Args:
        data_dir (str): The path to the root directory of the dataset.

    Returns:
        A pandas DataFrame with 'image_path' and one-hot encoded class columns.
    """
    image_paths = []
    classes = []

    # Define the specific directories (categories) you want to include
    target_categories = {'darkspots', 'clear_face', 'wrinkles', 'puffy_eyes'}

    # Walk through the directory structure
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        # Check if the path is a directory AND if its name is in our target list
        if os.path.isdir(class_path) and class_name in target_categories:
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image_paths.append(image_path)

                # Assign the single label for this image
                classes.append(class_name)

    # Create the initial DataFrame with single labels
    df = pd.DataFrame({
        'image_path': image_paths,
        'class': classes
    })

    # One-hot encode the 'class' column
    one_hot_encoded_df = pd.get_dummies(df, columns=['class'], prefix='', prefix_sep='')

    return one_hot_encoded_df


def create_age_df(data_dir, max_per_age=10):
    """
    Creates a pandas DataFrame with image paths and age labels, sampling
    up to a maximum number of images for each age. Ensures no duplicates.

    Args:
        data_dir (str): The path to the root directory containing the images.
                        Assumes filenames are in the format "age_something.format".
        max_per_age (int): The maximum number of images to sample per age group.

    Returns:
        A pandas DataFrame with 'image_path' and 'age' columns.
    """
    age_images = {}

    # Collect all image paths, grouped by age
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                age_str = filename.split('_')[0]
                age = int(age_str)
                if age not in age_images:
                    age_images[age] = []
                age_images[age].append(os.path.join(data_dir, filename))
            except (ValueError, IndexError):
                print(f"Skipping file: {filename} due to unexpected format.")
                continue

    final_paths = []
    final_ages = []

    # Sample images for each age
    for age, paths in age_images.items():
        # Shuffle the list of paths to randomize selection
        random.shuffle(paths)

        # Take the first `max_per_age` paths
        sampled_paths = paths[:max_per_age]

        # Extend the final lists
        final_paths.extend(sampled_paths)
        final_ages.extend([age] * len(sampled_paths))

    df = pd.DataFrame({
        'image_path': final_paths,
        'age': final_ages
    })

    return df


def train_test_split_custom(df, split_size=50):
    """
    Splits a DataFrame into a training set and a validation set.
    
    This function shuffles the DataFrame and reserves a fixed number of
    images for the validation set, with the rest for training.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        split_size (int): The number of images to reserve for the validation set.
        
    Returns:
        tuple: A tuple containing two DataFrames: (train_df, val_df).
               - train_df: The DataFrame for the training set.
               - val_df: The DataFrame for the validation set.
    """
    if len(df) < split_size:
        raise ValueError(f"The DataFrame must contain at least {split_size} images to create a validation set of that size.")

    # Resetting the index to make shuffling easier
    df = df.reset_index(drop=True)
    
    # Shuffle the DataFrame rows randomly
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select the first `split_size` rows for the validation set
    val_df = shuffled_df.iloc[:split_size]
    
    # The remaining rows are for the training set
    train_df = shuffled_df.iloc[split_size:]
    
    return train_df, val_df


def create_dual_head_model(input_shape=(224, 224, 3)):
    """
    Builds a dual-head model with an EfficientNetB0 backbone, a multi-label
    classification head for feature detection, and a regression head for age.
    
    Args:
        input_shape (tuple): The shape of the input images.
        
    Returns:
        A Keras Model object.
    """
    # Delete existing cached weights to force a fresh download
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'models')
    weights_path = os.path.join(cache_dir, 'efficientnetb0_notop.h5')
    if os.path.exists(weights_path):
        os.remove(weights_path)
    
    # Load the EfficientNetB0 backbone, pre-trained on ImageNet
    backbone = EfficientNetB0(
        weights='imagenet', # Keras handles the download automatically
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze the backbone layers to prevent them from being updated during training
    backbone.trainable = False
    
    inputs = Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Feature Detection Head (Multi-label Classification)
    feature_head_dense = Dense(128, activation='relu')(x)
    feature_head_dropout = Dropout(0.5)(feature_head_dense)
    # The number of neurons should match the number of one-hot encoded classes
    feature_detection_head = Dense(4, activation='sigmoid', name='feature_detection_head')(feature_head_dropout)
    
    # Age Prediction Head (Regression)
    age_prediction_head = Dense(1, activation='linear', name='age_prediction_head')(x)
    
    # Define the model with two outputs
    model = Model(
        inputs=inputs, 
        outputs=[feature_detection_head, age_prediction_head], 
        name="dual_head_model"
    )
    
    return model


def train_model(
    model, 
    train_dataset, 
    val_dataset,
    epochs=20
):
    """
    Compiles and trains the dual-head model with appropriate losses and metrics.
    
    Args:
        model (Model): The dual-head Keras model.
        train_dataset (tf.data.Dataset): The combined training dataset.
        val_dataset (tf.data.Dataset): The combined validation dataset.
        epochs (int): Number of training epochs.
    
    Returns:
        A tuple containing the trained model and its history object.
    """
    # Define loss functions for each head.
    losses = {
        'feature_detection_head': BinaryCrossentropy(),
        'age_prediction_head': MeanSquaredError()
    }

    # Define metrics for each head.
    metrics = {
        'feature_detection_head': ['binary_accuracy'],
        'age_prediction_head': ['mae']
    }

    # Compile the model with the Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics,
    )
    
    print("Training the model...")

    # Train the model with the validation set
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        verbose=1
    )
    
    print("Plotting training metrics...")
    
    # Plotting Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Total Training Loss')
    plt.plot(history.history['val_loss'], label='Total Validation Loss')
    plt.title('Total Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting Feature Detection Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['feature_detection_head_binary_accuracy'], label='Feature Head Training Accuracy')
    plt.plot(history.history['val_feature_detection_head_binary_accuracy'], label='Feature Head Validation Accuracy')
    plt.title('Feature Detection Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plotting Age Prediction MAE
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['age_prediction_head_mae'], label='Age Head Training MAE')
    plt.plot(history.history['val_age_prediction_head_mae'], label='Age Head Validation MAE')
    plt.title('Age Prediction Mean Absolute Error Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.show()

    return model, history


def df_to_dataset(df, label_cols, output_name, batch_size=32, shuffle=True):
    """
    Converts a pandas DataFrame into a tf.data.Dataset that yields (image, labels).

    Args:
        df (pd.DataFrame): Input DataFrame.
        label_cols (list): List of column names for labels.
        output_name (str): The name of the output head (e.g., 'age_prediction_head').
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle.

    Returns:
        tf.data.Dataset: Dataset yielding (image, label_dict).
    """
    if df.empty:
        return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)

    image_paths = df["image_path"].values
    labels = df[label_cols].values

    def load_image_and_labels(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0
        return img, {output_name: label}

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    """
    Orchestrates the entire machine learning pipeline.
    """
    print("Starting the facial and age prediction pipeline...")

    # Dataset directories
    FACIAL_DIR = r'D:\Projects\skin-age-detection\dataset_final'
    AGE_DIR = r'D:\Projects\skin-age-detection\UTKFace_resized'

    # Step 1: Model
    print("\nStep 1: Creating the dual-head model...")
    model = create_dual_head_model()
    model.summary()

    # Step 2: Load DataFrames
    print("\nStep 2: Loading and preprocessing data from disk...")
    try:
        age_df = create_age_df(AGE_DIR)
        onehot_df = create_onehot_facial_df(FACIAL_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your dataset directories.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return

    if age_df.empty and onehot_df.empty:
        print("Error: Both DataFrames are empty. Please check your dataset directories and file formats.")
        return

    # Step 3: Split into train/val
    print("\nStep 3: Splitting data into training and validation sets...")
    # Use split_size=50 for both splits, as specified in the original logic.
    age_data_train, age_data_val = train_test_split_custom(age_df, split_size=50)
    feature_data_train, feature_data_val = train_test_split_custom(onehot_df, split_size=50)

    print(f"Training age set size: {len(age_data_train)} samples")
    print(f"Validation age set size: {len(age_data_val)} samples")
    print(f"Training feature set size: {len(feature_data_train)} samples")
    print(f"Validation feature set size: {len(feature_data_val)} samples")

    # Step 4: Convert to tf.data.Dataset
    print("\nStep 4: Converting DataFrames to tf.data.Dataset...")
    batch_size = 32
    feature_cols = ["darkspots", "clear_face", "wrinkles", "puffy_eyes"]

    age_train_ds = df_to_dataset(age_data_train, ["age"], "age_prediction_head", batch_size=batch_size)
    age_val_ds = df_to_dataset(age_data_val, ["age"], "age_prediction_head", batch_size=batch_size, shuffle=False)

    feature_train_ds = df_to_dataset(feature_data_train, feature_cols, "feature_detection_head", batch_size=batch_size)
    feature_val_ds = df_to_dataset(feature_data_val, feature_cols, "feature_detection_head", batch_size=batch_size, shuffle=False)

    # Step 5: Combine datasets and train
    print("\nStep 5: Training the model...")
    
    # Merge the two datasets for a single training loop.
    # The 'labels' from each dataset are dictionaries that TensorFlow can match to the model heads.
    # We use 'concatenate' to stack the datasets for a unified `zip`.
    combined_train_ds = tf.data.Dataset.zip((age_train_ds.unbatch(), feature_train_ds.unbatch())).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    combined_val_ds = tf.data.Dataset.zip((age_val_ds.unbatch(), feature_val_ds.unbatch())).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    trained_model, history = train_model(
        model,
        combined_train_ds,
        combined_val_ds,
        epochs=20
    )

    print("\nPipeline complete. Model training finished.")

if __name__ == '__main__':
    main()