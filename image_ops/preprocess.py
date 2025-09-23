import os
import cv2
import numpy as np

# --- Define Haar Cascade path robustly ---
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
CASCADE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CASCADE_FILENAME)

# --- ImageNet mean and std for normalization ---
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# --- Preprocessing function ---
def preprocess_image(image_bytes):
    """
    Detects the primary face, crops, enhances, normalizes, resizes to 224x224,
    and returns the processed face. Returns None if no face is detected or if there's an error.
    """
    # 1. Validate cascade file
    if not os.path.exists(CASCADE_FILE_PATH):
        print(f"FATAL ERROR: Cascade file not found at '{CASCADE_FILE_PATH}'")
        return None

    face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)

    # 2. Decode image bytes
    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image_np is None:
        print("ERROR: Failed to decode image bytes.")
        return None

    # 3. Convert to grayscale and detect faces
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # 4. Crop first detected face
    x, y, w, h = faces[0]
    face_cropped = image_np[y:y+h, x:x+w]

    # 5. Enhance contrast using CLAHE
    gray_face = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray_face)
    face_enhanced = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    # 6. Resize to 224x224
    face_resized = cv2.resize(face_enhanced, (224, 224), interpolation=cv2.INTER_AREA)

    # 7. Normalize to [0,1] and apply ImageNet mean/std
    face_normalized = face_resized.astype(np.float32) / 255.0
    face_normalized = (face_normalized - MEAN) / STD

    return face_normalized


# --- Example usage block ---
if __name__ == '__main__':
    test_image_path = r"D:\Projects\skin-age-detection\datasets\UTKFace_resized\56_1_0_20170109220504650.jpg.chip.jpg"

    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at '{test_image_path}'")
    else:
        with open(test_image_path, 'rb') as f:
            image_bytes = f.read()

        print("Processing image...")
        processed_face = preprocess_image(image_bytes)

        if processed_face is not None:
            print("Face detected and processed successfully!")
            print(f"Shape of processed image: {processed_face.shape}")
            # Convert back to uint8 for visualization
            face_show = ((processed_face * STD + MEAN) * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imshow('Processed Face', face_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No face was detected in the provided image.")
