import os, random, shutil, zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2

# --- Parameters ---
DATASET_DIR = "/content/dataset_final"
WORK_DIR = "/content/dataset_capped"
DESIRED_CLASSES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
MAX_PER_CLASS = 1000
IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 20

os.makedirs(WORK_DIR, exist_ok=True)

# --- Step 1: Build class -> file path dictionary, cap at MAX_PER_CLASS ---
class_dict = {}
for cls in DESIRED_CLASSES:
    src_dir = os.path.join(DATASET_DIR, cls)
    imgs = [os.path.join(src_dir,f) for f in os.listdir(src_dir) if f.lower().endswith(("jpg","jpeg","png"))]
    random.shuffle(imgs)
    selected = imgs[:MAX_PER_CLASS]

    dst_dir = os.path.join(WORK_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)
    class_dict[cls] = []

    for f in selected:
        shutil.copy(f, dst_dir)
        class_dict[cls].append(os.path.join(dst_dir, os.path.basename(f)))

# --- Step 2: Split 80:20 per class ---
train_paths, val_paths, train_labels, val_labels = [], [], [], []

for idx, cls in enumerate(DESIRED_CLASSES):
    imgs = class_dict[cls]
    split_idx = int(len(imgs)*0.8)
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]

    train_paths.extend(train_imgs)
    train_labels.extend([idx]*len(train_imgs))
    val_paths.extend(val_imgs)
    val_labels.extend([idx]*len(val_imgs))

# --- Step 3: Shuffle combined train and val lists ---
train_combined = list(zip(train_paths, train_labels))
val_combined = list(zip(val_paths, val_labels))
random.shuffle(train_combined)
random.shuffle(val_combined)
train_paths, train_labels = zip(*train_combined)
val_paths, val_labels = zip(*val_combined)

# --- Step 4: Generator for on-the-fly loading ---
def data_generator(paths, labels, batch_size=BATCH_SIZE):
    n = len(paths)
    while True:
        for i in range(0, n, batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            X, y = [], []
            for p,l in zip(batch_paths, batch_labels):
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                img = preprocess_input(img.astype(np.float32))
                X.append(img)
                y.append(l)
            yield np.array(X), tf.keras.utils.to_categorical(y, num_classes=len(DESIRED_CLASSES))

train_gen = data_generator(train_paths, train_labels)
val_gen = data_generator(val_paths, val_labels)
steps_per_epoch = len(train_paths)//BATCH_SIZE
val_steps = len(val_paths)//BATCH_SIZE

# --- Step 5: Build and compile model ---
def build_model(num_classes):
    backbone = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    backbone.trainable = False  # freeze initially
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=backbone.input, outputs=preds)

model = build_model(len(DESIRED_CLASSES))
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# --- Callbacks ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# --- Step 6: Train ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# --- Step 7: Save model strictly as .h5 ---
model.save("efficientnet_b0_face.h5")
print("Model saved as efficientnet_b0_face.h5")
