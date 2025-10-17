# DermalScan: AI Facial Skin Aging Detection App

> A deep learning–based web app for detecting **facial skin conditions** and **estimating age** from images.  
> This project was developed as part of the **AI Facial Skin Aging Detection** initiative under **Infosys Springboard 6.0**.

---

## 🏫 Project Affiliation  

This project, **DermalScan: AI Facial Skin Aging Detection**, was developed as part of the  
**Infosys Springboard 6.0 Program** — an initiative to promote hands-on learning in Artificial Intelligence and Deep Learning.  

Under this program, the project integrates **computer vision** and **machine learning** concepts into a practical healthcare use case.  
It combines **dermatological image analysis** and **facial aging prediction** using advanced AI tools such as:  

- **TensorFlow / Keras (MobileNetV2)** for skin-condition classification  
- **OpenCV DNN** for face and age detection  
- **Streamlit** for deployment and visualization  

This project was mentored and executed under **Infosys Springboard’s AI/ML Learning Path**,  
with the goal of developing an **AI-driven diagnostic and skincare-assistance tool**.

---

## 🎯 Project Objective  

To identify facial aging indicators — such as **wrinkles, dark spots, puffy eyes, scars**, and **clear skin** — using computer vision and deep learning.  
It uses a combination of **MobileNetV2 (Keras)** for classification and **OpenCV DNN models** for multi-face detection and age estimation.

---

## 🧩 Key Features  

| Feature | Description |
|----------|--------------|
| 👥 **Multi-Face Detection** | Detects all faces in an image using OpenCV DNN |
| 🧴 **Skin-Condition Classification** | Predicts 6 dermal conditions with confidence scores |
| 📅 **Age Estimation** | Uses OpenCV AgeNet for apparent-age prediction |
| 💊 **Remedy Suggestions** | Natural & medical remedies per condition |
| 📊 **Results Export** | Download as CSV, annotated image, or combined ZIP |
| 💾 **SQLite Integration** | Stores past predictions |
| ♻️ **Reset Option** | Clears saved results & images |
| 🌐 **Interactive UI** | Responsive Streamlit interface |

---

## 🧩 Modules Overview  

| Module | Description | Tools |
|--------|--------------|-------|
| Dataset Setup | Six classes (Acne, Clear Face, Dark Spots, Puffy Eyes, Scars, Wrinkles) | Kaggle dataset |
| Preprocessing & Augmentation | Resizing, normalization, flips, rotations | `preprocess_augment.py` |
| Model Training | Transfer Learning with MobileNetV2 | TensorFlow / Keras |
| Face Detection & Prediction | Multi-face detection + condition & age prediction | OpenCV, TensorFlow |
| Frontend UI | Streamlit dashboard | Streamlit |
| Backend Integration | Connects preprocessing, model inference, output | Python |
| Exporting & Logging | Annotated image + CSV export | Pandas, zipfile |
| Database Integration | Stores prediction history | SQLite |
| Reset / Clear Feature | One-click cleanup | Streamlit + OS |

---

## 🧬 Tech Stack  

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | Python 3.10+ |
| Model | TensorFlow / Keras (MobileNetV2) |
| Face & Age Detection | OpenCV DNN (Caffe) |
| Data Preprocessing | NumPy, Pandas, PIL |
| Visualization & Export | OpenCV, zipfile, io |
| Database | SQLite |

---

## 📂 Project Structure  

DermalScan/
├── app.py
├── preprocess_augment.py
├── mobilenetv2_best_model.h5
├── age_prediction/
│ ├── age_deploy.prototxt
│ ├── age_net.caffemodel
│ ├── opencv_face_detector.pbtxt
│ └── opencv_face_detector_uint8.pb
├── output/
├── requirements.txt
└── README.md

## 🎯 Evaluation Criteria and Goals

The project modules are evaluated against specific metrics to ensure performance and usability:

| Focus Area | Metric / Evaluation Method | Target / Goal |
| :--- | :--- | :--- |
| **Data Preparation** | Dataset quality, augmentation effectiveness | Balanced & clean dataset. |
| **Model Performance** | Accuracy & loss metrics | **≥ 90% classification accuracy** and stable validation accuracy. |
| **UI & Backend** | Upload-to-output time & usability | **≤ 5 seconds per image**. |
| **Final Delivery** | Export functionality & documentation | Accurate export, log consistency, and professional completion. |
