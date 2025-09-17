# Import the necessary libraries
import os         # To interact with the operating system, especially for building file paths
import cv2        # OpenCV library for all computer vision tasks
import numpy as np    # NumPy library for numerical operations, especially with image arrays

# --- Define the path to the Haar Cascade XML file robustly ---
# __file__ is the path to the current script.
# os.path.dirname gets the directory of the script.
# os.path.join combines the directory and filename, making it work on any OS (Windows, Mac, Linux).
CASCADE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')

# --- Define the core preprocessing function ---
def preprocess_image(image_bytes):
    """
    Takes image bytes, detects the primary face, crops it, resizes it to 224x224,
    and returns the processed face image.

    Parameters:
    image_bytes (bytes): The raw byte content of the image file uploaded by the user.

    Returns:
    numpy.ndarray: The processed 224x224 face image in BGR format if a face is found.
    None: Returns None if no face is detected or if there's an error.
    """
    # 1. Check if the cascade file exists before trying to load it.
    if not os.path.exists(CASCADE_FILE_PATH):
        # Print an error to the console if the crucial XML file is missing.
        print(f"FATAL ERROR: Cascade file not found at {CASCADE_FILE_PATH}")
        # Return None to stop the process gracefully.
        return None

    # 2. Load the pre-trained Haar Cascade classifier from the defined path.
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)

    # 3. Decode the image bytes into a NumPy array that OpenCV can process.
    # The 'cv2.IMREAD_COLOR' flag ensures the image is loaded in its color format (BGR).
    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Check if the image was decoded successfully. If not, return None.
    if image_np is None:
        # This can happen if the image data is corrupted or in an unsupported format.
        return None

    # 4. Convert the color image to grayscale for the detection algorithm.
    # Haar Cascades are designed to work with single-channel grayscale images for speed and efficiency.
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 5. Perform face detection on the grayscale image.
    # This function returns a list of rectangles [x, y, width, height] for each face found.
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 6. Check if any faces were detected.
    if len(faces) == 0:
        # If the 'faces' list is empty, it means no faces were found.
        return None

    # 7. Extract the coordinates of the first face detected.
    # For simplicity and speed, we process only the first face in the list.
    x, y, w, h = faces[0]

    # 8. Crop the detected face from the original *color* image using array slicing.
    face_cropped = image_np[y:y+h, x:x+w]

    # 9. Resize the cropped face to the required 224x224 dimensions for the model.
    # 'cv2.INTER_AREA' is an efficient interpolation method for shrinking images.
    face_resized = cv2.resize(face_cropped, (224, 224), interpolation=cv2.INTER_AREA)

    # 10. Return the final, processed face image.
    return face_resized

# --- Example Usage Block (for testing the script directly) ---
if __name__ == '__main__':
    # This block runs only when you execute `python preprocess.py`.
    
    # IMPORTANT: Change this to the path of a test image on your computer.
    test_image_path = r'D:\Projects\skin-age-detection\datasets\utkface\115_1_1_20170112213257263.jpg.chip.jpg' 
    
    try:
        # Open and read the test image file in binary read mode ('rb').
        with open(test_image_path, 'rb') as f:
            # Read the entire content of the file into a bytes object.
            image_file_bytes = f.read()

        print("Attempting to process image...")
        # Call the preprocessing function with the image bytes.
        processed_face = preprocess_image(image_file_bytes)

        # Check the result of the function call.
        if processed_face is not None:
            # If a face was processed successfully, display it.
            print("Face detected and processed successfully!")
            print(f"Shape of processed image: {processed_face.shape}") # Should be (224, 224, 3).
            cv2.imshow('Processed Face', processed_face) # Display the image in a window.
            cv2.waitKey(0) # Wait for a key press before closing the window.
            cv2.destroyAllWindows() # Close all OpenCV windows.
        else:
            # If no face was found or an error occurred.
            print("No face was detected in the provided image.")

    except FileNotFoundError:
        # This error is caught if the test image path is incorrect.
        print(f"Error: Test image not found at '{test_image_path}'. Please check the path.")