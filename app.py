import streamlit as st
import tensorflow as tf
import numpy as np
import os
import dlib
import cv2
from PIL import Image
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Facial Analysis AI",
    page_icon="🤖",
    layout="wide"
)

# --- Helper File Downloader ---
def download_file(url, output_path):
    """Downloads a file from a URL if it doesn't exist."""
    if not os.path.exists(output_path):
        st.info(f"Downloading helper file: {output_path}...")
        r = requests.get(url, stream=True)
        with open(output_path, 'wb') as f:
            f.write(r.content)
        st.success(f"Downloaded {output_path}")

# --- Caching the Models (Load only once) ---
@st.cache_resource
def load_all_models_and_predictors(skin_model_path, age_model_path):
    """Loads all necessary models and detectors, decorated with Streamlit's caching."""
    # Download helper files for detectors
    download_file('https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt', 'deploy.prototxt')
    download_file('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
    download_file('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', 'shape_predictor_68_face_landmarks.dat.bz2')
    
    # Extract dlib model if not present
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(predictor_path):
        import bz2
        with bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2', 'rb') as f_in, open(predictor_path, 'wb') as f_out:
            f_out.write(f_in.read())

    # Load models
    face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    skin_model = tf.keras.models.load_model(skin_model_path, compile=False)
    age_model = tf.keras.models.load_model(age_model_path, compile=False)
    
    return face_net, landmark_predictor, skin_model, age_model

# --- The Backend Pipeline Function ---
def run_full_pipeline(image, face_net, landmark_predictor, skin_model, age_model):
    skin_class_names = ['clear face', 'dark spots', 'puffy eyes', 'wrinkles']
    IMG_SIZE = 224

    output_image = image.copy()
    (h, w) = image.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    best_detection_index = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, best_detection_index, 2]

    if confidence < 0.5:
        return image, {"error": "No face detected."}

    # Predictions
    box = detections[0, 0, best_detection_index, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    cropped_face = image[startY:endY, startX:endX]
    
    if cropped_face.size == 0:
        return image, {"error": "Face crop failed."}

    resized_face = tf.image.resize(cropped_face, [IMG_SIZE, IMG_SIZE])
    img_batch = np.expand_dims(resized_face, axis=0)
    preprocessed_img = tf.keras.applications.resnet_v2.preprocess_input(np.copy(img_batch))
    
    skin_probs = skin_model.predict(preprocessed_img)
    skin_class = skin_class_names[np.argmax(skin_probs[0])]
    skin_conf = np.max(skin_probs[0]) * 100
    
    age_pred = age_model.predict(preprocessed_img)[0][0]
    age_pred = max(0, age_pred)
    age_range_start = int(age_pred // 10) * 10
    age_range = f"{age_range_start}-{age_range_start + 10}"

    # Visualization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
    landmarks = landmark_predictor(gray, face_rect)
    
    if skin_class == 'puffy eyes' or skin_class == 'wrinkles':
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        cv2.rectangle(output_image, cv2.boundingRect(left_eye_pts), (0, 255, 255), 2)
        cv2.rectangle(output_image, cv2.boundingRect(right_eye_pts), (0, 255, 255), 2)
    
    cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Return the annotated image and a dictionary of results
    results = {
        "skin_class": skin_class,
        "skin_conf": skin_conf,
        "age_pred": age_pred,
        "age_range": age_range
    }
    return output_image, results

# --- Streamlit UI ---

st.title("🤖 AI Facial Analysis")
st.markdown("Upload an image to detect a face and predict skin condition and age.")

# Define model paths
SKIN_MODEL_PATH = 'model_skin.h5'
AGE_MODEL_PATH = 'age_resnet_simple_model_best_epoch5.h5'

# Load models with caching
try:
    face_net, landmark_predictor, skin_model, age_model = load_all_models_and_predictors(SKIN_MODEL_PATH, AGE_MODEL_PATH)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Face"):
        with st.spinner("Analyzing... This may take a moment."):
            # Run the full pipeline
            annotated_image, results = run_full_pipeline(image, face_net, landmark_predictor, skin_model, age_model)
            
            # Display results
            st.image(annotated_image, channels="BGR", caption="Analysis Result", use_column_width=True)
            
            st.subheader("Analysis Results")
            if "error" in results:
                st.error(results["error"])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Age", value=f"{results['age_pred']:.1f}", delta=f"{results['age_range']} range")
                with col2:
                    st.metric(label="Predicted Skin Condition", value=results['skin_class'], delta=f"{results['skin_conf']:.1f}% confidence")