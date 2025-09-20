import os  # Imports the 'os' module, mainly used here for path validation.
import cv2  # Imports the OpenCV library for computer vision tasks, like accessing the webcam.
import tensorflow as tf  # Imports the TensorFlow library for running the machine learning model.
import numpy as np  # Imports NumPy for numerical operations, especially for handling the video frames as arrays.

# --- Configuration ---
# A list of strings for the class names the model can predict. The order must match the model's output layer.
DESIRED_CLASSES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
# A tuple defining the target size (height, width) that the model expects for input images.
IMG_SIZE = (224, 224)
# ❗ IMPORTANT: Update this path to your trained model file.
MODEL_PATH = r"D:\Projects\skin-age-detection\models\mobilenet_effnet_head.h5"

def load_model_safe(model_path):
    """
    Safely loads a Keras model from the specified path with clear error handling.

    Parameters:
    model_path (str): The file path to the Keras model (.h5 file).

    Returns:
    tensorflow.keras.Model: The loaded Keras model object.
    """
    if not os.path.exists(model_path):  # Checks if a file exists at the given path.
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}\n👉 Please check the MODEL_PATH variable.")

    print(f"✅ Loading model from: {model_path}")  # Prints a confirmation message.
    return tf.keras.models.load_model(model_path)  # Loads the Keras model from the specified path and returns it.

# This block of code will only run when the script is executed directly.
if __name__ == "__main__":
    # --- 1. Load the Model Once ---
    model = load_model_safe(MODEL_PATH)  # Loads the model from disk into memory before starting the webcam.

    # --- 2. Initialize Webcam ---
    cap = cv2.VideoCapture(0)  # Initializes a video capture object. '0' refers to the default webcam.
    if not cap.isOpened():  # Checks if the webcam was successfully opened.
        raise IOError("❌ Cannot open webcam. Please check if it is connected and not in use by another application.")

    print("\n✅ Webcam initialized. Press 'q' to quit.") # Informs the user that the webcam is ready.

    # --- 3. Start Real-time Prediction Loop ---
    while True:  # Creates an infinite loop to continuously capture frames from the webcam.
        ret, frame = cap.read()  # Reads a single frame from the webcam. 'ret' is a boolean for success, 'frame' is the image.
        if not ret:  # If 'ret' is False, it means a frame could not be captured.
            print("❗️ Can't receive frame. Exiting ...") # Prints an error message.
            break  # Exits the loop.

        # --- 4. Preprocess the Frame for the Model ---
        # OpenCV captures in BGR format, but the model likely trained on RGB. We convert the color space.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # The frame must be resized to the size the model was trained on (224x224).
        resized_frame = cv2.resize(rgb_frame, IMG_SIZE)
        # Normalize the pixel values from the [0, 255] range to the [0.0, 1.0] range for the model.
        normalized_frame = resized_frame / 255.0
        # The model expects a "batch" of images. We add an extra dimension to our single frame to create a batch of 1.
        input_tensor = np.expand_dims(normalized_frame, axis=0)

        # --- 5. Make a Prediction ---
        preds = model.predict(input_tensor, verbose=0)[0]  # Passes the preprocessed frame to the model to get predictions.
        pred_idx = np.argmax(preds)  # Finds the index of the class with the highest probability score.
        pred_label = DESIRED_CLASSES[pred_idx]  # Retrieves the string label for the predicted class.
        confidence = preds[pred_idx] * 100  # Calculates the confidence percentage for the top prediction.

        # --- 6. Display the Results on the Frame ---
        # Create the text string to display on the video feed.
        display_text = f"{pred_label} ({confidence:.2f}%)"
        # Use OpenCV's putText function to draw the text onto the original frame.
        # Arguments: (image, text, origin_coordinates, font, font_scale, color, thickness)
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- 7. Show the Frame ---
        cv2.imshow('Live Face Analysis', frame)  # Displays the frame (with the text) in a window titled 'Live Face Analysis'.

        # --- 8. Check for Exit Key ---
        # Waits for 1 millisecond for a key press. `& 0xFF` is a standard bitmask.
        # If the key pressed is 'q' (for quit), the condition is met.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exits the infinite loop.

    # --- 9. Cleanup ---
    print("\n✅ Shutting down...") # Informs the user the program is closing.
    cap.release()  # Releases the webcam resource.
    cv2.destroyAllWindows()  # Closes all OpenCV windows that were created.