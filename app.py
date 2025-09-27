# app.py
import os
import streamlit as st
import cv2
import numpy as np
from datetime import datetime

from image_ops import loader, preprocess, label
from models import predict_age, predict_feature

# ==============================
# Root directories
# ==============================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CASCADE_DIR = os.path.join(ROOT_DIR, "image_ops")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Paths to files
FEATURES_MODEL_PATH = os.path.join(MODELS_DIR, r"D:\Projects\skin-age-detection\models\efficientnet_b0_face_classifier_finetuned.h5")
AGE_MODEL_PATH = os.path.join(MODELS_DIR, r"age_mobilenet_regression_stratified_old.h5")
CASCADE_FILENAME = os.path.join(CASCADE_DIR, r"haarcascade_frontalface_default.xml")

# ==============================
# Ensure logs folder exists
# ==============================
os.makedirs(LOGS_DIR, exist_ok=True)

# ==============================
# Cache model + cascade
# ==============================
@st.cache_resource
def load_models_and_cascade():
    face_cascade = loader.load_cascade(CASCADE_FILENAME)
    age_model = predict_age.load_model(AGE_MODEL_PATH)
    feature_model = predict_feature.load_model(FEATURES_MODEL_PATH)
    return face_cascade, age_model, feature_model

# ==============================
# Streamlit App
# ==============================
def main():
    st.set_page_config(page_title="Face Age & Feature Prediction", layout="wide")
    st.title("Facial Dermal Scan")

    with st.spinner("Loading models..."):
        face_cascade, age_model, feature_model = load_models_and_cascade()

    st.markdown(
        "Upload an image. Avoid accessories (glasses, earrings) "
        "and do not smile for best results."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    img_bytes = uploaded_file.read()

    # Decode original image
    full_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if full_image is None:
        st.error("Could not decode image.")
        return

    # Preprocessed cropped face for models
    try:
        face_image = preprocess.bytes_to_image(img_bytes)  # BGR (224x224)
    except Exception as e:
        st.error(f"Face preprocessing failed: {str(e)}")
        return

    with st.spinner("Predicting..."):
        try:
            # Predict age and features
            age = predict_age.predict_age(age_model, face_image)
            features = predict_feature.predict_features(feature_model, face_image)

            # Annotate full original image
            annotated = label.draw_labels_on_image(full_image.copy(), age, features, face_cascade)

        except Exception:
            st.error("No face detected or prediction failed. Try a well-lit image.")
            return

    # ==============================
    # Display predictions
    # ==============================
    st.subheader("Predictions")
    st.write(f"**Predicted Age:** {age:.1f} years")
    st.write("**Detected Features:**")
    for feat, prob in features.items():
        st.write(f"- {feat}: {prob*100:.1f}%")

    # Show only the annotated image (proper RGB)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption="Processed & Annotated Image", use_column_width=True)

    # ==============================
    # Downloads
    # ==============================

    # Annotated image as PNG (no double conversion)
    success, img_encoded = cv2.imencode(".png", annotated)
    if success:
        st.download_button(
            label="Download Annotated Image",
            data=img_encoded.tobytes(),
            file_name="annotated.png",
            mime="image/png"
        )

    # Predictions CSV
    csv_header = "timestamp,age," + ",".join(features.keys())
    csv_values = [datetime.now().isoformat(), f"{age:.1f}"] + [f"{prob*100:.1f}%" for prob in features.values()]
    csv_row = ",".join(csv_values)
    csv_text = csv_header + "\n" + csv_row

    # Save logs in logs/YYYY-MM-DD.csv
    log_filename = os.path.join(LOGS_DIR, f"predictions_{datetime.now().date()}.csv")
    if not os.path.exists(log_filename):
        with open(log_filename, "w") as f:
            f.write(csv_header + "\n")
    with open(log_filename, "a") as log_file:
        log_file.write(csv_row + "\n")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_text,
        file_name="predictions.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
