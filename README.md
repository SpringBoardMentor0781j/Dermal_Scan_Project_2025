# DermalScan: AI-Powered Facial Skin Aging Classifier and Age/Gender Predictor

This guide provides a comprehensive overview of the DermalScan project, covering its purpose, features, and instructions for both end-users and developers.

---

## 📌 Project Overview

DermalScan is a **deep learning-based system** designed to detect and classify facial aging signs (wrinkles, dark spots, puffy eyes) and predict a person's apparent **age** and **gender** from an image.  

The project leverages two distinct models:
- **Fine-tuned DenseNet121** → Classifier for identifying signs of aging.  
- **Custom CNN** → Performs dual task of **age regression** and **gender classification**.  

The backend integrates:
- Face detection (**Haar Cascades**)
- Image preprocessing
- Model inference  

A user-friendly **Flask web application** serves as the frontend, allowing users to upload images and view annotated results.

---

## ✨ Features

- **Face Detection** → Automatically detects and localizes human faces in an uploaded image.  
- **Aging Sign Classification** → Classifies detected faces into four categories:
  - Clear Face
  - Dark Spots
  - Puffy Eyes
  - Wrinkles  
  *(with confidence scores for each)*  
- **Age & Gender Prediction** → Predicts apparent **age** (integer) and **gender** ("Male" or "Female").  
- **Annotated Image Generation** → Creates annotated images with bounding boxes and predictions.  
- **Data Export** → Download annotated images and a CSV file with detailed predictions.  
- **Web Interface** → Simple, intuitive UI for uploads and results.  

---

## 🛠 Tech Stack

| Area        | Tools / Libraries |
|-------------|-------------------|
| Backend     | Python, Flask |
| Image Ops   | OpenCV, NumPy, Haar Cascades |
| Model       | TensorFlow/Keras, DenseNet121, Custom CNN |
| Datasets    | UTKFace (Age, Gender), Custom Dataset (Aging Signs) |
| Frontend    | HTML, CSS |
| Evaluation  | Accuracy, Loss, MAE, Confusion Matrix |
| Exporting   | CSV, Annotated Image |

---

## 👩‍💻 User Guide

### 🔎 How It Works
1. Upload an image through the web interface.  
2. Haar Cascade detects the **face location**.  
3. The detected face is cropped and preprocessed.  
4. **DenseNet121** → classifies aging signs.  
5. **Custom CNN** → predicts **age & gender**.  
6. The original image is annotated with predictions.  
7. Results displayed on the **results page** (with options to download).  

### 🌐 How to Use the Web App
1. Open the web app in your browser.  
2. Click **"Choose an Image"** → upload `.jpg`, `.jpeg`, or `.png`.  
3. Use a clear, well-lit face image for best results.  
4. Click **"Analyze Image"**.  
5. View the results: original + annotated image, predictions, confidence scores.  
6. Download annotated image or CSV report.  

---

## 👨‍💻 Developer Guide

### 📂 Project Structure
```
DermalScan/
│
├── app.py                                # Main Flask app
├── models/
│   ├── age_gender_model.h5               # Custom CNN (age/gender)
│   ├── fine_tuned_densenet_classifier.h5 # DenseNet (aging signs)
│   └── haarcascade_frontalface_default.xml # Haar Cascade (face detection)
│
├── static/
│   ├── css/
│   │   └── style.css                     # Web UI styling
│   └── uploads/                          # Uploaded & annotated images
│
├── templates/
│   ├── index.html                        # Upload form
│   └── results.html                      # Results page
│
├── age_gender_model_training_notebook.ipynb      # CNN training notebook
└── densenet_final_model_training_notebook.ipynb  # DenseNet training notebook
```

---

### ⚙️ Setup and Installation

#### 1. Prerequisites
- Python **3.8+**
- `pip`

#### 2. Clone the Repository
```bash
git clone https://github.com/your-username/DermalScan.git
cd DermalScan
```

#### 3. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scriptsctivate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 4. Install Dependencies
Create a **requirements.txt**:
```
Flask
tensorflow
opencv-python
numpy
pandas
werkzeug
scikit-learn
seaborn
matplotlib
tqdm
Pillow
```

Install:
```bash
pip install -r requirements.txt
```

#### 5. Place Models
Download/pretrained models and place inside `models/`:
- `age_gender_model.h5`
- `fine_tuned_densenet_classifier.h5`
- `haarcascade_frontalface_default.xml`

---

### ▶️ Running the Application
```bash
flask run
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📊 Model Information

### 1. Aging Sign Classifier
- **Model**: DenseNet121 (ImageNet pre-trained, fine-tuned)  
- **Task**: Multi-class classification  
- **Dataset**: Custom dataset with 4 categories  
- **Performance Target**: ≥ 88% accuracy  

### 2. Age & Gender Predictor
- **Model**: Custom CNN  
- **Tasks**:
  - Age → regression  
  - Gender → classification  
- **Dataset**: UTKFace  
- **Performance Targets**:
  - Age: MAE < 4 years  
  - Gender: Accuracy > 90%  

---

## 🔬 Replicating Model Training

### 📓 DenseNet Training
Notebook: `densenet_final_model_training_notebook.ipynb`  
- Setup: Download custom dataset → update `DATASET_PATH`  
- Process: Data preprocessing, augmentation, transfer learning, fine-tuning  

### 📓 Age/Gender CNN Training
Notebook: `age_gender_model_training_notebook.ipynb`  
- Setup: Download UTKFace → update `BASE_DIR`  
- Process: Label parsing, preprocessing (128×128 grayscale), CNN training (multi-output)  

---

## ✅ Summary
DermalScan provides an **end-to-end ML pipeline** for:
- Detecting facial regions  
- Classifying aging signs  
- Predicting **age & gender**  
- Annotating and exporting results  

The combination of **DenseNet121** + **Custom CNN** makes DermalScan a robust tool for dermatological and demographic analysis.  

---
