# Import the necessary libraries
import cv2        # OpenCV for computer vision tasks
import numpy as np  # NumPy for numerical operations

# --- NEW: Import the custom loader function ---
# This line imports the 'load_cascade' function from your 'loader.py' file.
from loader import load_cascade

def draw_labels_on_image(image_np, age, features, face_cascade):
    """
    Pads an image, detects a face, and draws the predicted age and feature percentages.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array (in BGR format).
        age (float): The predicted age of the person.
        features (dict): A dictionary of feature names (str) and their probabilities (float).
        face_cascade (cv2.CascadeClassifier): The pre-loaded Haar Cascade classifier object.

    Returns:
        numpy.ndarray: The padded image with the annotations drawn on it.
    """
    # --- 1. Add Padding to the Image ---
    # Define the size of the border to add around the image.
    top_pad = 70      # Space above for the age text.
    bottom_pad = 150  # Space below for the list of features.
    left_pad = 50     # Aesthetic space on the sides.
    right_pad = 50    # Aesthetic space on the sides.
    border_color = [0, 0, 0] # The color of the border (black).

    # Use OpenCV's copyMakeBorder function to create a new, larger image with the padding.
    output_image = cv2.copyMakeBorder(image_np, top_pad, bottom_pad, left_pad, right_pad, 
                                      cv2.BORDER_CONSTANT, value=border_color)

    # 2. Convert the padded color image to grayscale for the detection algorithm.
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    # 3. Perform face detection on the new, padded grayscale image.
    # The (x, y) coordinates will be relative to this new, larger canvas.
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # 4. If no faces are found, return the padded image without annotations.
    if len(faces) == 0:
        print("DEBUG: No face was detected in the image.")
        return output_image

    # 5. Process the first face detected.
    x, y, w, h = faces[0]

    # --- Draw Annotations on the Padded Image ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    age_color = (255, 255, 255); feature_color = (0, 255, 0); box_color = (255, 0, 0)
    font_scale = 0.6; thickness = 2

    # 6. Draw a bounding box around the detected face.
    cv2.rectangle(output_image, (x, y), (x+w, y+h), box_color, thickness)

    # 7. Draw the predicted age above the bounding box.
    # Because we padded the image, there is now guaranteed space at (y - 10).
    age_text = f"Age: {age:.1f} years"
    cv2.putText(output_image, age_text, (x, y - 10), font, font_scale, age_color, thickness)

    # 8. Draw the feature percentages below the bounding box.
    # There is also guaranteed space for this list now.
    start_y_features = y + h + 25
    for i, (feature_name, probability) in enumerate(features.items()):
        feature_text = f"- {feature_name}: {probability * 100:.1f}%"
        current_y = start_y_features + (i * 25)
        cv2.putText(output_image, feature_text, (x, current_y), font, font_scale, feature_color, 1)

    # 9. Return the final padded and annotated image.
    return output_image

# --- Example Usage Block (for testing the script directly) ---
if __name__ == '__main__':
    # Define the path to a test image on your computer.
    test_image_path = r'D:\Projects\skin-age-detection\datasets\UTKFace_resized\40_0_1_20170113184933016.jpg.chip.jpg'
    
    # Use the loader module to get the classifier.
    cascade_classifier = load_cascade()
    
    if cascade_classifier is not None:
        # Create dummy prediction data.
        dummy_age = 38.4
        dummy_features = {"Wrinkles": 0.87, "Puffy Eyes": 0.62, "Dark Spots": 0.25}
        
        # Read the test image.
        image = cv2.imread(test_image_path)
        
        if image is not None:
            # Pass the original image to the function. It will handle the padding.
            labeled_image = draw_labels_on_image(image, dummy_age, dummy_features, cascade_classifier)
            
            # Display the final image.
            cv2.imshow("Labeled Output with Padding", labeled_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Error: Could not read the image from '{test_image_path}'.")
    else:
        print("Could not run labeling because the cascade classifier failed to load.")