import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Load Models ---
try:
    age_gender_model = load_model('models/age_gender_model.h5', compile=False)
    age_gender_model.compile(
        optimizer='adam',
        loss={'gender_out': 'binary_crossentropy', 'age_out': 'mae'},
        metrics={'gender_out': 'accuracy', 'age_out': 'mae'}
    )

    aging_signs_model = load_model('models/best_densenet_classifier.h5')
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    logging.info("Models loaded and compiled successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    age_gender_model = None
    aging_signs_model = None
    face_cascade = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # --- Image Processing and Prediction ---
            if age_gender_model and aging_signs_model and face_cascade:
                try:
                    # Load image with OpenCV (loads as BGR)
                    image = cv2.imread(filepath)
                    
                    # *** FIX: Convert image from BGR to RGB for Keras model compatibility ***
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

                    if len(faces) == 0:
                        return render_template('results.html', error="No face detected in the image.")

                    (x, y, w, h) = faces[0]

                    # --- Aging Signs Prediction (DenseNet121) ---
                    # *** FIX: Use the RGB image for preprocessing ***
                    face_roi_signs = rgb_image[y:y+h, x:x+w]
                    face_roi_signs = cv2.resize(face_roi_signs, (224, 224))
                    face_roi_signs = face_roi_signs.astype("float") / 255.0
                    face_roi_signs = img_to_array(face_roi_signs)
                    face_roi_signs = np.expand_dims(face_roi_signs, axis=0)
                    
                    aging_preds = aging_signs_model.predict(face_roi_signs)[0]
                    aging_signs = {
                        "clear_face": aging_preds[0],
                        "dark_spots": aging_preds[1],
                        "puffy_eyes": aging_preds[2],
                        "wrinkles": aging_preds[3]
                    }
                    top_aging_sign = max(aging_signs, key=aging_signs.get)

                    # --- Age and Gender Prediction ---
                    face_roi_age_gender = gray[y:y+h, x:x+w]
                    face_roi_age_gender = cv2.resize(face_roi_age_gender, (128, 128))
                    face_roi_age_gender = face_roi_age_gender.astype("float") / 255.0
                    face_roi_age_gender = img_to_array(face_roi_age_gender)
                    face_roi_age_gender = np.expand_dims(face_roi_age_gender, axis=0)

                    gender_pred, age_pred = age_gender_model.predict(face_roi_age_gender)
                    gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
                    age = int(age_pred[0][0])

                    # --- Annotate Image ---
                    # Use the original BGR image for annotation with OpenCV
                    annotated_image = image.copy()
                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    annotations = [f"{k.replace('_', ' ').title()}: {v:.1%}" for k, v in aging_signs.items()]
                    annotations.extend([f"Age: {age}", f"Gender: {gender}"])

                    y_offset = y - 10 if y - 10 > 10 else y + h + 20
                    for i, text in enumerate(annotations):
                        cv2.putText(annotated_image, text, (x, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    annotated_filename = f"annotated_{filename}"
                    annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
                    cv2.imwrite(annotated_filepath, annotated_image)

                    # --- Prepare CSV ---
                    csv_data = {
                        "Feature": list(aging_signs.keys()) + ["age", "gender"],
                        "Prediction": [f"{v:.2%}" for v in aging_signs.values()] + [age, gender]
                    }
                    df = pd.DataFrame(csv_data)
                    csv_filename = f"predictions_{filename.rsplit('.', 1)[0]}.csv"
                    csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
                    df.to_csv(csv_filepath, index=False)

                    # --- Analysis Message ---
                    if top_aging_sign in ['dark_spots', 'puffy_eyes', 'wrinkles']:
                        analysis_message = f"Major Aging Feature Found: {top_aging_sign.replace('_', ' ').title()}"
                    else:
                        analysis_message = "Little to no signs of aging found. The face appears clear."

                    return render_template('results.html',
                                           original_image=filename,
                                           annotated_image=annotated_filename,
                                           predictions=aging_signs,
                                           age=age,
                                           gender=gender,
                                           csv_file=csv_filename,
                                           analysis_message=analysis_message)
                except Exception as ex:
                    logging.error(f"An error occurred during prediction: {ex}")
                    return render_template('results.html', error="An error occurred during analysis.")
            else:
                return render_template('results.html', error="Models are not loaded. Please check the server logs.")

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)