import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import pandas as pd
from datetime import datetime


st.set_page_config(page_title="AI DermalScan", layout="centered")


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("model", exist_ok=True)


LABELS = ["wrinkles", "darkspots", "puffy_eyes", "clear_skin"]


@st.cache_resource
def load_model(model_path: str = "model/model.h5"):
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
    except Exception as e:
        st.warning(f"Could not load saved model at {model_path}: {e}")

    # Build a small classifier on top of EfficientNetB0 (imagenet weights)
    try:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3)
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        out = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        st.info("Using EfficientNetB0 base model with random classifier head (no fine-tuned weights found).")
        return model
    except Exception as e:
        st.error(f"Unable to create model: {e}")
        raise


def detect_faces(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def preprocess_face(face_rgb):
    # input: RGB numpy array of face
    img = cv2.resize(face_rgb, (224, 224))
    arr = tf.keras.applications.efficientnet.preprocess_input(img.astype("float32"))
    return np.expand_dims(arr, axis=0)


def annotate_image(image_np, faces, predictions):
    # image_np is RGB
    out = image_np.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(out, (x, y), (x + w, y + h), (10, 90, 180), 2)  # blue-ish
        # write top label with highest class
        if i < len(predictions):
            probs = predictions[i]
            top_idx = int(np.argmax(probs))
            label = LABELS[top_idx]
            text = f"{label}: {probs[top_idx]:.2f}"
            cv2.putText(out, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 90, 180), 2)
    return out


def save_csv_log(filename, timestamp, faces, predictions):
    ensure_dirs()
    csv_path = os.path.join("outputs", "predictions.csv")
    rows = []
    if len(faces) == 0:
        rows.append({
            "filename": filename,
            "timestamp": timestamp,
            "face_index": None,
            **{f"prob_{lab}": None for lab in LABELS},
        })
    else:
        for i, probs in enumerate(predictions):
            row = {"filename": filename, "timestamp": timestamp, "face_index": i}
            for j, lab in enumerate(LABELS):
                row[f"prob_{lab}"] = float(probs[j])
            rows.append(row)

    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    ensure_dirs()
    st.markdown("<h1 style='color:#0b4f8a'>AI DermalScan: Facial Skin Aging Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#145c9e'>Upload a face image and the app will detect faces and predict skin aging indicators.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Options")
        st.write("Model: EfficientNetB0-based classifier")
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        st.markdown("---")
        st.write("Theme: Blue / Off-white")
        st.write("Outputs saved to `outputs/predictions.csv`")

    model = load_model()

    if uploaded_file is None:
        st.info("Please upload an image to analyze.")
        return

    # read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read image. Make sure file is a valid image.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detect_faces(img_rgb)
    st.write(f"Detected {len(faces)} face(s)")

    predictions = []
    for (x, y, w, h) in faces:
        # include small padding
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_rgb.shape[1], x + w + pad)
        y2 = min(img_rgb.shape[0], y + h + pad)
        face_crop = img_rgb[y1:y2, x1:x2]
        try:
            pre = preprocess_face(face_crop)
            probs = model.predict(pre)[0]
        except Exception as e:
            st.warning(f"Prediction failed for a face: {e}")
            probs = np.array([0.0] * len(LABELS))
        predictions.append(probs)

    annotated = annotate_image(img_rgb, faces, predictions)
    st.image(annotated, caption="Annotated image", use_column_width=True)

    # Show detailed probabilities
    if len(predictions) > 0:
        all_rows = []
        for i, probs in enumerate(predictions):
            row = {"face_index": i}
            for j, lab in enumerate(LABELS):
                row[lab] = float(probs[j])
            all_rows.append(row)
        df = pd.DataFrame(all_rows)
        st.subheader("Per-face probabilities")
        st.dataframe(df.style.format({lab: "{:.3f}" for lab in LABELS}))
        st.bar_chart(df.set_index("face_index"))
    else:
        st.info("No faces found to run predictions on.")

    # Download annotated image
    pil_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download annotated image", buf.getvalue(), file_name="annotated.png", mime="image/png")

    # Save CSV log
    timestamp = datetime.utcnow().isoformat()
    csv_path = save_csv_log(uploaded_file.name or "uploaded_image", timestamp, faces, predictions)
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    st.download_button("Download predictions CSV", csv_bytes, file_name=os.path.basename(csv_path), mime="text/csv")


if __name__ == "__main__":
    main()
