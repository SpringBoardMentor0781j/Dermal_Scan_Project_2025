import streamlit as st
import numpy as np
import cv2
from fpdf import FPDF
import tempfile
import os

from image_ops.preprocess import preprocess_image
from image_ops.label import draw_labels_on_image
from models.predict_age import load_age_model, predict_age
from models.predict_feature import load_model_safe, predict_image_array

# --- caching models and cascade ---
@st.cache_resource
def get_models():
    age_model = load_age_model("models/age_mobilenet_regression_stratified.h5")
    feature_model = load_model_safe("models/mobilenet_effnet_head.h5")
    face_cascade = draw_labels_on_image.__globals__['load_cascade']()
    return age_model, feature_model, face_cascade

age_model, feature_model, face_cascade = get_models()

st.title("Face Age & Feature Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # preprocess
    processed_face = preprocess_image(file_bytes)
    
    if processed_face is None:
        st.error("No face detected in the uploaded image.")
    else:
        age_pred = predict_age(age_model, processed_face)
        features_pred = predict_image_array(processed_face/255.0, feature_model)
        feature_names = ["clear_face", "darkspots", "puffy_eyes", "wrinkles"]
        features_dict = {name: float(features_pred[i]) for i, name in enumerate(feature_names)}
        
        annotated = draw_labels_on_image(processed_face, age_pred, features_dict, face_cascade)
        
        st.image(annotated, channels="BGR", caption="Prediction Result")
        
        # --- PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(0, 10, f"Predicted Age: {age_pred:.1f} years", ln=1)
        pdf.cell(0, 10, "Features:", ln=1)
        for k,v in features_dict.items():
            pdf.cell(0, 8, f"- {k}: {v*100:.1f}%", ln=1)

        # save annotated image temporarily
        temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_img_file.name, annotated)
        pdf.image(temp_img_file.name, x=10, y=50, w=pdf.w-20)
        temp_img_file.close()
        
        # save PDF to temporary file
        temp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf_file.name)
        temp_pdf_file.close()
        
        with open(temp_pdf_file.name, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="report.pdf", mime="application/pdf")
        
        # cleanup temp files
        os.remove(temp_img_file.name)
        os.remove(temp_pdf_file.name)
