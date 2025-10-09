# DermalScan: AI Facial Skin Aging Detection App

DermalScan is an intelligent system designed to analyze facial images and identify common indicators of skin aging. This deep learning-based application leverages a pre-trained **EfficientNetB0** model, fine-tuned to classify features into four specific categories: **wrinkles, dark spots, puffy eyes, and clear skin**.

## 🌟 Project Statement and Objective

The core objective of DermalScan is to develop a robust deep learning system that can detect and classify various signs of facial aging. The application provides a user-friendly, web-based frontend where users can upload images and visualize the detected aging signs, complete with annotated bounding boxes and labels.

## ✨ Key Features and Outcomes

DermalScan provides a complete pipeline from image upload to detailed analysis:

*   **Facial Feature Detection:** The system detects and localizes facial features that indicate aging. This process utilizes **OpenCV** and **Haar Cascades** for initial face detection.
*   **Classification:** Features are classified into four main categories: **wrinkles, dark spots, puffy eyes, and clear skin**.
*   **Deep Learning Core:** The system uses a **pretrained EfficientNetB0** model for transfer learning and robust classification.
*   **Detailed Predictions:** Outputs include displaying predictions as **percentages** (class probability) and classifying the detected features using the trained Convolutional Neural Network (CNN) model.
*   **Web Interface (UI):** A web-based frontend, potentially developed using **Streamlit** or HTML/CSS, handles image uploads and provides an output preview.
*   **Visualization:** Results are visualized with **annotated bounding boxes** and labels detailing the class probability.
*   **Data Export:** The application allows users to download the annotated image and export prediction logs, potentially in **CSV** format, for documentation or analysis.

## 🛠️ Tech Stack and Dependencies

The DermalScan application relies on a modular set of tools for computer vision, machine learning, and web deployment:

| Area | Tools / Libraries | Details |
| :--- | :--- | :--- |
| **Model** | **EfficientNetB0**, **TensorFlow/Keras** | EfficientNetB0 is a Keras image classification model often loaded with weights pre-trained on ImageNet. |
| **Image Ops** | **OpenCV**, **NumPy**, **Haarcascade** | Used for face detection, image processing, and numerical operations. |
| **Frontend** | **Streamlit** or **HTML, CSS** | Provides the Web UI for image upload and result display. |
| **Backend** | **Python**, Modularized Inference | Handles the processing pipeline and model inference.
| **Exporting** | **CSV**, Annotated Image, PDF (optional) | Used for generating final logs and outputs. |

## ⚙️ Model Details and Preprocessing

The model training and deployment follow specific guidelines:

1.  **Transfer Learning:** The project utilizes the pretrained **EfficientNetB0** architecture for transfer learning. EfficientNet models expect inputs to be float tensors of pixels with values in the **** range.
2.  **Training Configuration:** The model is trained using **categorical cross-entropy loss** and the **Adam optimizer**.
3.  **Input Preparation:** Input images are resized and normalized to **224x224** pixels.
4.  **Data Augmentation:** To enhance robustness, image augmentation techniques such as flip, rotation, and zoom are applied. Class labels are processed using one-hot encoding.
5.  **Face Detection:** Before classification, Haar Cascade is used to detect faces, and the model is applied specifically to these cropped face regions.

## 📁 Project Structure

The project is structured into several key directories and files necessary for training, inference, and deployment:

```
/
├── data_preprocessing/     # Dataset Setup, labeling, augmentation, and encoding scripts
├── model_trainer/          # Scripts for Model Training with EfficientNetB0
├── models/                 # Stores the trained CNN model (.h5 file)
├── image_ops/              # Scripts for Face Detection (Haar Cascade) and prediction pipeline
├── logs/                   # Stores prediction logs and bounding box data
├── docs/                   # Documentation and guides (including this README)
├── .gitignore
└── app.py                  # Main Python script for the Streamlit/Backend Web UI
```

## 🎯 Evaluation Criteria and Goals

The project modules are evaluated against specific metrics to ensure performance and usability:

| Focus Area | Metric / Evaluation Method | Target / Goal |
| :--- | :--- | :--- |
| **Data Preparation** | Dataset quality, augmentation effectiveness | Balanced & clean dataset. |
| **Model Performance** | Accuracy & loss metrics | **≥ 90% classification accuracy** and stable validation accuracy. |
| **UI & Backend** | Upload-to-output time & usability | **≤ 5 seconds per image**. |
| **Final Delivery** | Export functionality & documentation | Accurate export, log consistency, and professional completion. |