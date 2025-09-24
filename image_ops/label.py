import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from models import predict_age
# ==============================
# Load Haar Cascade
# ==============================
def load_cascade(cascade_filename="haarcascade_frontalface_default.xml"):
    cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename)
    if not os.path.exists(cascade_path):
        print(f"FATAL ERROR: Cascade file not found at '{cascade_path}'")
        return None

    cascade_classifier = cv2.CascadeClassifier(cascade_path)
    if cascade_classifier.empty():
        print(f"FATAL ERROR: Could not load cascade from '{cascade_path}'. File may be corrupt.")
        return None

    print(f"Haar Cascade '{cascade_filename}' loaded successfully.")
    return cascade_classifier


# ==============================
# Load Models (inference only)
# ==============================
def load_models():
    age_model = load_model(
        r"D:\Projects\skin-age-detection\models\age_mobilenet_regression_stratified_old.h5",
        compile=False
    )
    features_model = load_model(
        r"D:\Projects\skin-age-detection\models\mobilenet_effnet_head.h5",
        compile=False
    )
    print("Models loaded successfully (inference only).")
    return age_model, features_model

 
# ==============================
# Predict Age and Features
# ==============================
def predict_age_and_features(face_img, age_model, features_model, feature_names):
    face_resized = cv2.resize(face_img, (224, 224))
    face_norm = face_resized.astype("float32") / 255.0
    face_input = np.expand_dims(face_norm, axis=0)

    # Age prediction
    age_pred = age_model.predict(face_input)
    if age_pred.shape[-1] > 1:  # classification-style
        predicted_age = np.argmax(age_pred, axis=1)[0]
    else:  # regression-style
        predicted_age = age_pred[0][0]

    # Features prediction
    feature_probs = features_model.predict(face_input)[0]
    features = dict(zip(feature_names, feature_probs))

    return predicted_age, features


# ==============================
# Draw Labels on Image
# ==============================
def draw_labels_on_image(image_np, age, features, face_cascade):
    top_pad, bottom_pad, left_pad, right_pad = 70, 150, 50, 50
    border_color = [0, 0, 0]
    output_image = cv2.copyMakeBorder(
        image_np, top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_CONSTANT, value=border_color
    )

    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("DEBUG: no face detected in image")
        return output_image

    x, y, w, h = faces[0]
    x -= left_pad
    y -= top_pad
    x, y = max(0, x), max(0, y)

    font = cv2.FONT_HERSHEY_SIMPLEX
    age_color, feature_color, box_color = (255, 255, 255), (0, 255, 0), (255, 0, 0)
    font_scale, thickness = 0.6, 2

    cv2.rectangle(output_image, (x + left_pad, y + top_pad), (x + w + left_pad, y + h + top_pad), box_color, thickness)

    age_text = f"Age: {age:.1f} years"
    cv2.putText(output_image, age_text, (x + left_pad, y + top_pad - 10), font, font_scale, age_color, thickness)

    start_y_features = y + h + top_pad + 25
    for i, (feature_name, probability) in enumerate(features.items()):
        feature_text = f"- {feature_name}: {probability*100:.1f}%"
        current_y = start_y_features + (i * 25)
        cv2.putText(output_image, feature_text, (x + left_pad, current_y), font, font_scale, feature_color, 1)

    return output_image


# ==============================
# Main
# ==============================
if __name__ == '__main__':
    test_image_path = r"C:\Users\Hi\Downloads\WhatsApp Image 2025-09-22 at 3.07.08 PM.jpeg"
    cascade_classifier = load_cascade()

    if cascade_classifier is not None:
        age_model, features_model = load_models()
        feature_names = ["Wrinkles", "Puffy Eyes", "Dark Spots"]  # match your model output

        image = cv2.imread(test_image_path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade_classifier.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_crop = image[y:y+h, x:x+w]

                predicted_age, predicted_features = predict_age_and_features(
                    face_crop, age_model, features_model, feature_names
                )
                predicted_age= predict_age.predict_age
                labeled_image = draw_labels_on_image(image, predicted_age, predicted_features, cascade_classifier)
                cv2.imshow("Labeled Output with Padding", labeled_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No face detected in image.")
        else:
            print(f"Error: could not read image from '{test_image_path}'")
    else:
        print("Cascade classifier failed to load, stopping.")
