import os
from tensorflow.keras.models import load_model

def load_keras_model(model_path: str):
    """
    Load a Keras .h5 model for inference only.

    Args:
        model_path (str): Full path to the model file.

    Returns:
        tf.keras.Model: The loaded model object.

    Raises:
        FileNotFoundError: If the given model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")
    return model
