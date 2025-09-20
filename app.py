import streamlit as st
import numpy as np
import cv2
import tempfile
from fpdf import FPDF

from image_ops.preprocess import preprocess_image
from image_ops.label import draw_labels_on_image, load_cascade
from models.predict_age import load_age_model, predict_age
from models.predict_feature import load_model_safe, predict_image_array


# Cache model + cascade loading so it happens only once
@st.cache_resource
def get_models():
    age_model = load_age_model("models/age_mobilenet_regression_stratified.h5")
    feature_model = load_model_safe("models/mobilenet_effnet_head.h5")
    face_cascade = load_cascade()
    return age_model, feature_model, face_cascade


age_model, feature_model, face_cascade = get_models()

st.title("Face Age & Feature Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # decode uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read uploaded image.")
    else:
        # preprocess for model input (expects image array, not bytes)
        processed_face = preprocess_image(img)

        if processed_face is None:
            st.error("No face detected in the uploaded image.")
        else:
            # predictions
            age_pred = predict_age(age_model, processed_face)
            features_pred = predict_image_array(processed_face / 255.0, feature_model)

            feature_names = ["clear_face", "darkspots", "puffy_eyes", "wrinkles"]
            features_dict = {
                name: float(features_pred[i]) for i, name in enumerate(feature_names)
            }

            # annotate original image
            annotated = draw_labels_on_image(img, age_pred, features_dict, face_cascade)

            # display in Streamlit (convert BGR→RGB for correct colors)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Prediction Result")

            # create PDF report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(0, 10, f"Predicted Age: {age_pred:.1f} years", ln=1)
            pdf.cell(0, 10, "Features:", ln=1)
            for k, v in features_dict.items():
                pdf.cell(0, 8, f"- {k}: {v*100:.1f}%", ln=1)

            # save annotated image temporarily and embed in PDF
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                cv2.imwrite(tmp_img.name, annotated)
                pdf.image(tmp_img.name, x=10, y=50, w=pdf.w - 20)

            # save PDF to temp file for download
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                pdf.output(tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as f:
                    st.download_button(
                        "Download PDF Report",
                        f,
                        file_name="report.pdf",
                        mime="application/pdf"
                    )
