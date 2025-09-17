import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Config ---
DESIRED_CLASSES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
IMG_SIZE = (224, 224)

def load_model_safe(model_path="mobilenet_effnet_head.h5"):
    """Safely loads a Keras model with error handling."""
    if not os.path.isabs(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model file not found at: {model_path}\n"
            f"👉 Place the model in this folder or give absolute path."
        )

    print(f"✅ Loading model from: {model_path}")
    return tf.keras.models.load_model(model_path)

def predict_image(image_path, model_path="mobilenet_effnet_head.h5"):
    """
    Loads a trained model and predicts probabilities for all classes.
    Displays the image and a bar chart of class probabilities.
    """
    # Load model
    model = load_model_safe(model_path)

    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array_exp, verbose=0)[0]

    # Get best prediction
    pred_idx = np.argmax(preds)
    pred_label = DESIRED_CLASSES[pred_idx]
    confidence = preds[pred_idx] * 100

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(f"Predicted: {pred_label}\n({confidence:.2f}%)")

    # Bar chart of probabilities
    y_pos = np.arange(len(DESIRED_CLASSES))
    axes[1].barh(y_pos, preds * 100, align='center', color="skyblue")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(DESIRED_CLASSES)
    axes[1].invert_yaxis()  # highest prob at top
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Class Probabilities")

    # Highlight predicted class
    axes[1].barh(pred_idx, preds[pred_idx] * 100, color="green")

    plt.tight_layout()
    plt.show()

    return pred_label, confidence, preds

if __name__ == "__main__":
    # Example usage
    test_image = r"D:\Projects\skin-age-detection\datasets\utkface\86_1_0_20170120225751953.jpg.chip.jpg"
    pred_label, confidence, preds = predict_image(
        test_image,
        model_path=r"mobilenet_effnet_head.h5"
    )
    print(f"\n✅ Final Prediction -> {pred_label}: {confidence:.2f}%")
