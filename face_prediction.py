import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------
# Paths
# ---------------------------
FACE_PROTO = r"C:\Dermal scan\age_prediction\opencv_face_detector.pbtxt"
FACE_MODEL = r"C:\Dermal scan\age_prediction\opencv_face_detector_uint8.pb"
AGE_PROTO = r"C:\Dermal scan\age_prediction\age_deploy.prototxt"
AGE_MODEL = r"C:\Dermal scan\age_prediction\age_net.caffemodel"
DERMAL_MODEL = r"C:\Dermal scan\mobilenetv2_best_model.h5"
OUTPUT_DIR = r"C:\Dermal scan\output"

# ---------------------------
# Configs
# ---------------------------
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_BUCKET_CENTERS = np.array([1.0, 5.0, 10.0, 17.5,
                               28.5, 40.5, 50.5, 80.0])
CLASS_NAMES = ["Acne", "Clear Face", "Dark Spots", "Puffy Eyes", "Scars", "Wrinkles"]

# ---------------------------
# Load Models
# ---------------------------
face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
dermal_model = load_model(DERMAL_MODEL)

print("✅ Models loaded successfully")

# ---------------------------
# Age Correction Function
# ---------------------------
def corrected_age(age_cont, dermal_label, dermal_conf):
    if dermal_label.lower() == "wrinkles":
        if dermal_conf >= 0.95:
            return max(age_cont, 75.0)
        elif dermal_conf >= 0.90:
            return max(age_cont, 65.0)
        elif dermal_conf >= 0.80:
            return max(age_cont, 55.0)
    return age_cont

# ---------------------------
# Process Image
# ---------------------------
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image from {image_path}")
        return

    # Add border so labels don’t get cut
    border_size = 80
    image = cv2.copyMakeBorder(image, border_size, border_size,
                               border_size, border_size,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            face_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not face_boxes:
        print("⚠ No faces detected.")
        return

    for (x1, y1, x2, y2) in face_boxes:
        face = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        # --- Dermal Prediction ---
        dermal_input = cv2.resize(face, (224, 224))
        dermal_input = preprocess_input(dermal_input.astype(np.float32))
        dermal_input = np.expand_dims(dermal_input, axis=0)
        preds = dermal_model.predict(dermal_input)
        class_idx = preds[0].argmax()
        dermal_class = CLASS_NAMES[class_idx]
        dermal_conf = preds[0][class_idx]

        # --- Age Prediction ---
        age_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(age_blob)
        age_probs = age_net.forward()[0]
        age_cont = float(np.sum(age_probs * AGE_BUCKET_CENTERS))
        max_age_conf = float(age_probs.max())
        age_cont = corrected_age(age_cont, dermal_class, dermal_conf)
        bucket_idx = int(np.argmin(np.abs(AGE_BUCKET_CENTERS - age_cont)))
        age_bucket = AGE_BUCKETS[bucket_idx]

        # Final labels
        dermal_label = f"{dermal_class}: {dermal_conf*100:.1f}%"
        age_label = f"Age: {age_bucket} ({age_cont:.1f} yrs)"
        if max_age_conf < 0.45:
            age_label += " [Uncertain]"

        # Overlay labels
        offset = 25
        cv2.putText(image, dermal_label, (x1, y2 + offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, age_label, (x1, y2 + offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Save and show
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_DIR, f"annotated_{base_name}")
    cv2.imwrite(save_path, image)
    print(f"💾 Saved annotated result to: {save_path}")

    cv2.imshow("Dermal + Age Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------
# ---------------------------
# Main
# ---------------------------
def main():
    test_image = r"C:\Dermal scan\OIP.webp"
    process_image(test_image)

if __name__ == "__main__":
    main()
