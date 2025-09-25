# app.py
import os
import streamlit as st
import cv2
import numpy as np

from image_ops import loader, preprocess, label
from models import predict_age, predict_feature

# ==============================
# Root directories
# ==============================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
CASCADE_DIR = os.path.join(ROOT_DIR, "image_ops")

# Paths to files
FEATURES_MODEL_PATH = os.path.join(MODELS_DIR, "D:\Projects\skin-age-detection\models\efficientnet_b0_face_classifier_finetuned.h5")
AGE_MODEL_PATH = os.path.join(MODELS_DIR, "age_mobilenet_regression_stratified_old.h5")
CASCADE_FILENAME = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")

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
    st.title("👤 Face Age & Feature Prediction")

    st.markdown("Upload a face image to predict **age** and **visible features**.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        image_np = preprocess.bytes_to_image(img_bytes)  # BGR (H,W,3)

        st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB),
                 caption="Uploaded Image", use_column_width=True)

        with st.spinner("Loading models..."):
            face_cascade, age_model, feature_model = load_models_and_cascade()

        with st.spinner("Predicting..."):
            # Age
            age = predict_age.predict_age(age_model, image_np)

            # Features
            features = predict_feature.predict_features(feature_model, image_np)

            # Annotated image
            annotated = label.draw_labels_on_image(image_np.copy(), age, features, face_cascade)

        st.subheader("Predictions")
        st.write(f"**Predicted Age:** {age:.1f} years")
        st.write("**Detected Features:**")
        for feat, prob in features.items():
            st.write(f"- {feat}: {prob*100:.1f}%")

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption="Annotated Output", use_column_width=True)


if __name__ == "__main__":
    main()
