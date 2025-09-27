# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================

import os                   # For interacting with the operating system (e.g., creating directories, file paths).
import sys                  # For system-specific functions, like exiting the script.
import zipfile              # For extracting files from a .zip archive.
import shutil               # For high-level file operations like copying.
import random               # For shuffling data to ensure randomness.
import numpy as np          # For numerical operations, especially for arrays and matrices.
import tensorflow as tf     # The core deep learning library.
from google.colab import drive # To mount and access your Google Drive within Colab.
import cv2                  # OpenCV library for image processing tasks.

# --- Import specific Keras modules from TensorFlow ---
from tensorflow.keras.applications import EfficientNetB0                 # The pre-trained EfficientNetB0 model.
from tensorflow.keras.applications.efficientnet import preprocess_input  # The specific preprocessing function for EfficientNet.
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # Layers for building the custom classifier head.
from tensorflow.keras.models import Model                                # The base class for creating a Keras model.
from tensorflow.keras.optimizers import Adam                             # The Adam optimizer for training the model.
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Callbacks to improve the training process.


# ==============================================================================
# SECTION 2: PARAMETERS
# ==============================================================================
ZIP_FILE_NAME = "dataset_final.zip"
LOCAL_EXTRACT_PATH = "/content/dataset_unzipped"
EXTRACT_DIR = os.path.join(LOCAL_EXTRACT_PATH, "dataset_final")
WORK_DIR = "/content/dataset_cleaned"
DESIRED_CLASSES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
MAX_PER_CLASS = 1000
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
VAL_SPLIT = 0.2
INITIAL_EPOCHS = 10
FINETUNE_EPOCHS = 15


# ==============================================================================
# SECTION 3: DATA PREPARATION
# ==============================================================================

# --- Mount Google Drive ---
print("🚀 Mounting Google Drive...")
drive.mount('/content/drive')

# --- Step 1: Find, Copy, and Extract the Dataset Zip ---
def find_copy_and_extract_zip(zip_name, search_dir='/content/drive/MyDrive', local_copy_path='/content', extract_to_path='/content/dataset_unzipped'):
    print(f"🔎 Searching for '{zip_name}' in '{search_dir}'...")
    zip_path_in_drive = None
    for root, dirs, files in os.walk(search_dir):
        if zip_name in files:
            zip_path_in_drive = os.path.join(root, zip_name)
            break
    if zip_path_in_drive is None:
        raise FileNotFoundError(f"'{zip_name}' not found in '{search_dir}'.")
    print(f"✅ Found file at: {zip_path_in_drive}")
    local_zip_path = os.path.join(local_copy_path, zip_name)
    print(f"🚚 Copying '{zip_name}' to local Colab environment...")
    shutil.copy(zip_path_in_drive, local_zip_path)
    print("✅ Copying complete.")
    os.makedirs(extract_to_path, exist_ok=True)
    print(f"📦 Extracting '{zip_name}' to '{extract_to_path}'...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print("✅ Extraction complete.")
    return extract_to_path
try:
    find_copy_and_extract_zip(ZIP_FILE_NAME, extract_to_path=LOCAL_EXTRACT_PATH)
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    sys.exit()

# --- Step 2: Cap and Copy to Working Directory ---
print("🚚 Capping and copying dataset...")
os.makedirs(WORK_DIR, exist_ok=True)
for cls in DESIRED_CLASSES:
    src_dir = os.path.join(EXTRACT_DIR, cls)
    if not os.path.exists(src_dir):
        print(f"⚠️ Warning: Source directory not found: {src_dir}. Skipping.")
        continue
    imgs = [f for f in os.listdir(src_dir) if f.lower().endswith(("jpg", "jpeg", "png"))]
    random.shuffle(imgs)
    selected = imgs[:MAX_PER_CLASS]
    dst_dir = os.path.join(WORK_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)
    for img_name in selected:
        shutil.copy(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))
print("✅ Capping and copying complete.")

# --- Step 3: Validate, Clean, and prepare file lists ---
def validate_clean_and_list_files(target_dir, class_map):
    print(f"🧼 Validating and cleaning images in '{target_dir}'...")
    valid_filepaths, valid_labels = [], []
    for class_name, class_idx in class_map.items():
        class_dir = os.path.join(target_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(class_dir, filename)
                try:
                    img = cv2.imread(filepath)
                    if img is None:
                        os.remove(filepath)
                        continue
                    channels = img.shape[2] if len(img.shape) == 3 else 1
                    if channels != 3:
                        bgr_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if channels == 1 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        cv2.imwrite(filepath, bgr_image)
                    valid_filepaths.append(filepath)
                    valid_labels.append(class_idx)
                except Exception:
                    os.remove(filepath)
    print("✅ Cleaning and list creation complete.")
    return valid_filepaths, valid_labels
class_to_idx = {name: i for i, name in enumerate(DESIRED_CLASSES)}
all_filepaths, all_labels = validate_clean_and_list_files(WORK_DIR, class_to_idx)

# --- Step 4: Final Sanity Check and Data Splitting ---
if not all_filepaths:
    print("\n❌ CRITICAL ERROR: No valid image files remain after the cleaning process.")
    sys.exit()
else:
    print(f"✅ Sanity check passed. Found {len(all_filepaths)} valid image files.")
print("🔀 Splitting data...")
combined = list(zip(all_filepaths, all_labels))
random.shuffle(combined)
all_filepaths, all_labels = zip(*combined)
split_idx = int(len(all_filepaths) * (1 - VAL_SPLIT))
train_paths, val_paths = all_filepaths[:split_idx], all_filepaths[split_idx:]
train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
print(f"📊 Data split complete. Training: {len(train_paths)}, Validation: {len(val_paths)}")


# ==============================================================================
# SECTION 4: CREATE TF.DATA DATASET
# ==============================================================================
print("🚀 Building tf.data pipelines...")
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])
    image = preprocess_input(image)
    label = tf.one_hot(label, depth=len(DESIRED_CLASSES))
    return image, label
train_ds = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
train_ds = train_ds.shuffle(buffer_size=len(train_paths)).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
print("✅ tf.data pipelines are ready.")

# ==============================================================================
# SECTION 5: BUILD THE MODEL (MODIFIED)
# ==============================================================================
def build_model(num_classes):
    """
    Builds the model and now returns BOTH the final model and the backbone object.
    """
    # Define the base model (EfficientNetB0) and assign it to the 'backbone' variable.
    backbone = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # Start with the backbone frozen.
    backbone.trainable = False
    # Create the new custom classifier head.
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    # Create the final model object.
    model = Model(inputs=backbone.input, outputs=preds)
    # Return both the final model and the backbone so we can control it later.
    return model, backbone

print("🛠️ Building the model...")
# Capture both the model and the backbone object when calling the function.
model, backbone = build_model(len(DESIRED_CLASSES))

# ==============================================================================
# SECTION 6: TWO-STAGE TRAINING (MODIFIED)
# ==============================================================================
print("\n--- STAGE 1: Training the Classifier Head ---")
initial_lr = 0.0005
model.compile(optimizer=Adam(learning_rate=initial_lr), loss="categorical_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, callbacks=[early_stop, reduce_lr])

print("\n--- STAGE 2: Fine-Tuning the Top Layers ---")
# --- CORRECTED CODE ---
# Use the 'backbone' variable directly to unfreeze its layers.
backbone.trainable = True
# Define how many layers from the end to unfreeze.
fine_tune_at = -30
# Loop through the bottom layers of the backbone and keep them frozen.
for layer in backbone.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a very low learning rate for fine-tuning.
finetune_lr = 0.00001
model.compile(optimizer=Adam(learning_rate=finetune_lr), loss="categorical_crossentropy", metrics=["accuracy"])
print(f"Unfrozen layers for fine-tuning: {len(model.trainable_variables)}")

# Correctly set the initial epoch for the continuation of training.
last_epoch = history.epoch[-1] + 1 if history.epoch else 0
# Continue training the model.
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=last_epoch,
    epochs=last_epoch + FINETUNE_EPOCHS, # Train up to a total number of epochs.
    callbacks=[early_stop, reduce_lr]
)

# ==============================================================================
# SECTION 7: SAVE THE FINAL MODEL
# ==============================================================================
print("💾 Saving the final, fine-tuned model...")
model.save("efficientnet_b0_face_classifier_finetuned.h5")
print("🎉 Model saved as efficientnet_b0_face_classifier_finetuned.h5")