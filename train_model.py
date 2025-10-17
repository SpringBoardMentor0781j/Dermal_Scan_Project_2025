import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# -----------------------------
# Paths to .npy files
# -----------------------------
X_path = r"C:\Dermal scan\augmented_X.npy"
y_path = r"C:\Dermal scan\augmented_y.npy"

# -----------------------------
# Check if files exist
# -----------------------------
if not os.path.exists(X_path):
    raise FileNotFoundError(f"File not found: {X_path}")
if not os.path.exists(y_path):
    raise FileNotFoundError(f"File not found: {y_path}")

print("✅ .npy files found. Loading dataset...")

# -----------------------------
# Load dataset
# -----------------------------
X = np.load(X_path)
y = np.load(y_path)
print(f"Dataset loaded: {X.shape}, {y.shape}")

# -----------------------------
# Train-validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y.argmax(axis=1)
)

# -----------------------------
# Compute class weights
# -----------------------------
y_labels = y_train.argmax(axis=1)
class_weights = compute_class_weight("balanced", classes=np.unique(y_labels), y=y_labels)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# Build MobileNetV2 Model
# -----------------------------
inputs = Input(shape=(224,224,3))
base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(y.shape[1], activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
print("✅ Model built successfully")

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint("mobilenetv2_best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

# -----------------------------
# Stage 1: Train Frozen Base
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[checkpoint, lr_scheduler, early_stop]
)

# -----------------------------
# Stage 2: Fine-tuning last 30 layers
# -----------------------------
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_ft = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[checkpoint, lr_scheduler, early_stop]
)

# -----------------------------
# Plot Accuracy and Loss
# -----------------------------
def plot_metrics(history1, history2=None):
    acc = history1.history['accuracy']
    val_acc = history1.history['val_accuracy']
    loss = history1.history['loss']
    val_loss = history1.history['val_loss']

    if history2:
        acc += history2.history['accuracy']
        val_acc += history2.history['val_accuracy']
        loss += history2.history['loss']
        val_loss += history2.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(12,5))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.show()

plot_metrics(history, history_ft)
