# import necessary libraries
import cv2        # opencv for computer vision tasks
import numpy as np  # numpy for numerical operations
import os
# --- the custom loader function ---


def load_cascade(cascade_filename="haarcascade_frontalface_default.xml"):
    """
    Loads a Haar Cascade classifier from a file.

    Args:
        cascade_filename (str): The name of the cascade XML file. It's assumed
                                to be in the same directory as this script.

    Returns:
        cv2.CascadeClassifier or None: The loaded classifier object if successful,
                                       otherwise None.
    """
    # 1. Create a robust, absolute path to the cascade file.
    # This joins the directory of the current script with the filename,
    # ensuring the file is found regardless of where the script is run from.
    cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename)

    # 2. Check if the cascade file actually exists at the constructed path.
    if not os.path.exists(cascade_path):
        # If the file is not found, print a clear error message.
        print(f"FATAL ERROR: Cascade file not found at '{cascade_path}'")
        # Return None to indicate that the loading failed.
        return None

    # 3. Load the classifier from the file path.
    cascade_classifier = cv2.CascadeClassifier(cascade_path)

    # 4. Check if the classifier was loaded correctly (i.e., it's not empty).
    # A corrupted file might exist but fail to load into a valid classifier.
    if cascade_classifier.empty():
        # If loading resulted in an empty object, print an error.
        print(f"FATAL ERROR: Could not load cascade classifier from '{cascade_path}'. The file might be corrupt.")
        # Return None to indicate failure.
        return None

    # 5. If all checks pass, print a success message and return the classifier object.
    print(f"Haar Cascade classifier '{cascade_filename}' loaded successfully.")
    return cascade_classifier



def draw_labels_on_image(image_np, age, features, face_cascade):
    """
    pads an image, detects a face, and draws the predicted age and feature percentages

    Args:
        image_np (numpy.ndarray): input image as numpy array (bgr format)
        age (float): predicted age
        features (dict): feature names (str) -> probabilities (float)
        face_cascade (cv2.CascadeClassifier): preloaded haar cascade

    Returns:
        numpy.ndarray: padded image with annotations
    """
    # --- 1. add padding to image ---
    top_pad, bottom_pad, left_pad, right_pad = 70, 150, 50, 50
    border_color = [0,0,0] # black

    output_image = cv2.copyMakeBorder(image_np, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=border_color)

    # 2. convert to grayscale for detection
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    # 3. detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # 4. no faces found
    if len(faces) == 0:
        print("DEBUG: no face detected in image")
        return output_image

    # 5. process first face detected
    x, y, w, h = faces[0]

    # --- fix offset: adjust coords by padding ---
    x -= left_pad
    y -= top_pad
    x = max(0, x); y = max(0, y)  # make sure coords don't go negative

    # --- draw annotations ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    age_color, feature_color, box_color = (255,255,255), (0,255,0), (255,0,0)
    font_scale, thickness = 0.6, 2

    # 6. draw bounding box (adjusted for padding)
    cv2.rectangle(output_image, (x+left_pad, y+top_pad), (x+w+left_pad, y+h+top_pad), box_color, thickness)

    # 7. draw age text above box
    age_text = f"Age: {age:.1f} years"
    cv2.putText(output_image, age_text, (x+left_pad, y+top_pad - 10), font, font_scale, age_color, thickness)

    # 8. draw feature percentages below box
    start_y_features = y + h + top_pad + 25
    for i, (feature_name, probability) in enumerate(features.items()):
        feature_text = f"- {feature_name}: {probability*100:.1f}%"
        current_y = start_y_features + (i*25)
        cv2.putText(output_image, feature_text, (x+left_pad, current_y), font, font_scale, feature_color, 1)

    # 9. return final image
    return output_image

# --- example usage ---
if __name__ == '__main__':
    test_image_path = r'D:\Projects\skin-age-detection\datasets\UTKFace_10k_balanced\90_1_2_20170111221639268.jpg.chip.jpg'
    cascade_classifier = load_cascade()

    if cascade_classifier is not None:
        dummy_age = 38.4
        dummy_features = {"Wrinkles": 0.87, "Puffy Eyes": 0.62, "Dark Spots": 0.25}

        image = cv2.imread(test_image_path)
        if image is not None:
            labeled_image = draw_labels_on_image(image, dummy_age, dummy_features, cascade_classifier)
            cv2.imshow("Labeled Output with Padding", labeled_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"error: could not read image from '{test_image_path}'")
    else:
        print("could not run labeling because cascade classifier failed to load")
