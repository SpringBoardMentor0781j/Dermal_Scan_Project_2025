


# Import necessary libraries
import cv2  # OpenCV for loading the cascade classifier
import os   # To build file paths robustly and check for file existence

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

# --- Example Usage Block (for testing the loader directly) ---
if __name__ == '__main__':
    # This block runs only when you execute `python loader.py`.
    
    print("--- Testing the cascade loader ---")
    
    # Call the function to load the classifier.
    face_detector = load_cascade()
    
    # Check the result.
    if face_detector is not None:
        # If the returned object is not None, it means loading was successful.
        print("✅ Test successful. The cascade object is ready to use.")
    else:
        # If the object is None, it means there was an error during loading.
        print("❌ Test failed. Please check the error messages above.")