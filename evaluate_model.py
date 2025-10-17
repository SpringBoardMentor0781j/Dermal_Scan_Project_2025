# evaluate_model.py

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
X_path = r"C:\Dermal scan\augmented_X.npy"
y_path = r"C:\Dermal scan\augmented_y.npy"
model_path = r"C:\Dermal scan\mobilenetv2_best_model.h5"

# -----------------------------
# Load dataset
# -----------------------------
print("📂 Loading dataset...")
X = np.load(X_path)
y = np.load(y_path)
print(f"✅ Dataset loaded: {X.shape}, {y.shape}")

# Same split as training
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y.argmax(axis=1)
)

# -----------------------------
# Load trained model
# -----------------------------
print("📂 Loading trained model...")
model = tf.keras.models.load_model(model_path)
print(f"✅ Loaded model from {model_path}")

# -----------------------------
# Evaluate model
# -----------------------------
loss, acc = model.evaluate(X_val, y_val, verbose=1)
print(f"📊 Validation Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# -----------------------------
# Predictions
# -----------------------------
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

class_names = ["acne", "clear_face", "dark_spots", "puffy_eyes", "scars", "wrinkles"]

# Classification report
print("\n📑 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
