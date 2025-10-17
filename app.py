import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import pandas as pd
from PIL import Image
import zipfile
import io

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="DermalScan: AI Facial Skin Aging", layout="wide")

# ---------------------------
# Logo + Title
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    logo = Image.open(r"C:\Users\Duggineni upendar\Downloads\icon 2.jpg")
except FileNotFoundError:
    logo = None

col1, col2 = st.columns([0.08, 1])
with col1:
    if logo:
        st.image(logo, width=70)
with col2:
    st.markdown(
        "<h1 style='display:inline; margin:0; padding:0'>DermalScan: AI Facial Skin Aging Detection</h1>",
        unsafe_allow_html=True
    )

st.markdown(
    """
    Detect and classify **facial aging signs** such as wrinkles, dark spots, puffy eyes, scars, and clear skin.  
    Upload an image to visualize aging signs with annotated bounding boxes, confidence scores, and recommended remedies.
    """,
    unsafe_allow_html=True
)
st.write("---")

# ---------------------------
# Paths & Configs
# ---------------------------
FACE_PROTO = r"C:\Dermal scan\age_prediction\opencv_face_detector.pbtxt"
FACE_MODEL = r"C:\Dermal scan\age_prediction\opencv_face_detector_uint8.pb"
AGE_PROTO = r"C:\Dermal scan\age_prediction\age_deploy.prototxt"
AGE_MODEL = r"C:\Dermal scan\age_prediction\age_net.caffemodel"
DERMAL_MODEL = r"C:\Dermal scan\mobilenetv2_best_model.h5"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_BUCKET_CENTERS = np.array([1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0])
CLASS_NAMES = ["Acne", "Clear Face", "Dark Spots", "Puffy Eyes", "Scars", "Wrinkles"]

OUTPUT_DIR = r"C:\Dermal scan\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Remedies
# ---------------------------
REMEDIES = {
    "Acne": {
        "Natural": ["Honey mask", "Green tea compress", "Aloe vera gel"],
        "Medical": ["Topical benzoyl peroxide", "Retinoid cream", "Antibiotic cream"]
    },
    "Wrinkles": {
        "Natural": ["Coconut oil massage", "Aloe vera gel", "Vitamin E application"],
        "Medical": ["Retinol cream", "Chemical peels", "Botox injections"]
    },
    "Dark Spots": {
        "Natural": ["Lemon juice", "Aloe vera gel", "Green tea extract"],
        "Medical": ["Vitamin C serum", "Hydroquinone cream", "Chemical exfoliants"]
    },
    "Puffy Eyes": {
        "Natural": ["Cold tea bags", "Cucumber slices", "Aloe vera gel"],
        "Medical": ["Eye creams with caffeine", "Lymphatic drainage massage"]
    },
    "Scars": {
        "Natural": ["Aloe vera gel", "Honey", "Coconut oil massage"],
        "Medical": ["Silicone sheets", "Laser therapy", "Microneedling"]
    },
    "Clear Face": {
        "Natural": ["Maintain hydration", "Balanced diet"],
        "Medical": ["Continue current skincare routine"]
    }
}

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_all_models():
    face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    dermal_model = load_model(DERMAL_MODEL)
    return face_net, age_net, dermal_model

face_net, age_net, dermal_model = load_all_models()
st.success("✅ Models loaded successfully!")

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_uploaded_image(image, target_max_dim=600, target_min_dim=200):
    h, w = image.shape[:2]
    if max(h, w) > target_max_dim:
        scale = target_max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    elif min(h, w) < target_min_dim:
        scale = target_min_dim / min(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

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
# Annotate Image
# ---------------------------
def annotate_image(image, confidence_thresh=0.5, border_size=20):
    if image is None:
        return None, None, None

    # Convert to BGR (3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                               cv2.BORDER_CONSTANT, value=[0, 0, 0])
    h, w = image.shape[:2]

    # Face detection (DNN + NMS)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes, confidences = [], []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence_thresh:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, 0.4)
    faces = []
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        box = boxes[i]
        x1, y1, w_box, h_box = box
        faces.append((x1, y1, x1 + w_box, y1 + h_box, confidences[i]))

    # Haar fallback
    if not faces:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for (x, y, w_box, h_box) in haar.detectMultiScale(gray, 1.1, 5):
            faces.append((x, y, x + w_box, y + h_box, 0.5))

    if not faces:
        st.warning("⚠ No faces detected.")
        return image, None, None

    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,255),(255,128,0)]
    details = []

    for idx, (x1, y1, x2, y2, conf) in enumerate(faces):
        color = colors[idx % len(colors)]
        face = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if face.size == 0:
            continue

        # Dermal Prediction
        dermal_input = cv2.resize(face, (224,224))
        dermal_input = preprocess_input(dermal_input.astype(np.float32))
        dermal_input = np.expand_dims(dermal_input, axis=0)
        preds = dermal_model.predict(dermal_input, verbose=0)
        class_idx = preds[0].argmax()
        dermal_class = CLASS_NAMES[class_idx]
        dermal_conf = preds[0][class_idx]

        # Age Prediction
        age_blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(age_blob)
        age_probs = age_net.forward()[0]
        age_cont = corrected_age(float(np.sum(age_probs * AGE_BUCKET_CENTERS)), dermal_class, dermal_conf)
        bucket_idx = int(np.argmin(np.abs(AGE_BUCKET_CENTERS - age_cont)))
        age_bucket = AGE_BUCKETS[bucket_idx]
        max_age_conf = float(age_probs.max())

        dermal_label = f"{dermal_class}: {dermal_conf*100:.1f}%"
        age_label = f"Age: {age_bucket} ({age_cont:.1f} yrs)"
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 25

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"Face #{idx+1}", (x1,label_y), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        cv2.putText(image, dermal_label, (x1,y2+25), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        cv2.putText(image, age_label, (x1,y2+50), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        details.append({
            "Face ID": idx+1,
            "Face Confidence": f"{conf*100:.1f}%",
            "Dermal Class": dermal_class,
            "Dermal Confidence": f"{dermal_conf*100:.1f}%",
            "Predicted Age": age_bucket,
            "Age (Years)": f"{age_cont:.1f}",
            "Age Confidence": f"{max_age_conf*100:.1f}%",
            "Bounding Box": f"{x1},{y1},{x2},{y2}",
            "Natural Remedies": ", ".join(REMEDIES.get(dermal_class, {}).get("Natural", [])),
            "Medical Remedies": ", ".join(REMEDIES.get(dermal_class, {}).get("Medical", []))
        })

    save_path = os.path.join(OUTPUT_DIR, "annotated_uploaded_image.jpg")
    cv2.imwrite(save_path, image)
    return image, save_path, details

# ---------------------------
# Sidebar: Person Info + Upload
# ---------------------------
st.sidebar.header("Person Information")
person_name = st.sidebar.text_input("Name")
person_id = st.sidebar.text_input("ID (optional)")
person_age = st.sidebar.number_input("Age", min_value=0, max_value=120)
person_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
notes = st.sidebar.text_area("Notes / Comments")

uploaded_file = st.sidebar.file_uploader("Upload a Face Image", type=["jpg","jpeg","png","webp"])

# ---------------------------
# Main Logic
# ---------------------------
if uploaded_file is not None and person_name:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Invalid image.")
    else:
        image = preprocess_uploaded_image(image)
        with st.spinner("Analyzing..."):
            annotated_image, save_path, details = annotate_image(image)

        if annotated_image is not None:
            st.subheader("Annotated Result")
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        if details:
            for row in details:
                row.update({
                    "Name": person_name,
                    "ID": person_id,
                    "Person Age": person_age,
                    "Gender": person_gender,
                    "Notes": notes
                })

            st.subheader("Prediction Details (All Faces)")
            df = pd.DataFrame(details)
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode("utf-8")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zipf:
                zipf.writestr("multi_face_predictions.csv", csv_data)
                zipf.write(save_path, arcname="annotated_image.jpg")
            zip_buffer.seek(0)

            st.download_button(
                "📦 Download All (CSV + Annotated Image)",
                data=zip_buffer.getvalue(),
                file_name="dermalscan_results.zip",
                mime="application/zip"
            )
            st.download_button(
                "📊 Download CSV Only",
                data=csv_data,
                file_name="multi_face_predictions.csv",
                mime="text/csv"
            )
            st.download_button(
                "🖼 Download Annotated Image Only",
                data=open(save_path, "rb"),
                file_name="annotated_image.jpg",
                mime="image/jpeg"
            )
