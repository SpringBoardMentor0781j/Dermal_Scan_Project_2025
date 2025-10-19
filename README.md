DermalScan: AI-Powered Facial Skin Aging Classifier and Age Predictor
This guide provides a comprehensive overview of the DermalScan project, covering its purpose, features, and instructions for both end-users and developers.
________________________________________
📌 Project Overview
DermalScan is a deep learning-based system designed to detect and classify facial aging signs (wrinkles, dark spots, puffy eyes) and predict a person's apparent age  from an image.
The project leverages two distinct models:
•	ResNet50V2→ Classifier for identifying signs of aging. 
•	MobileNetV2 → Performs dual task of age regression. 
The backend integrates:
•	Face detection (Haar Cascades) 
•	Image preprocessing 
•	Model inference 
A user-friendly Streamlit web application serves as the frontend, allowing users to upload images and view annotated results.
________________________________________
✨ Features
•	Face Detection → Automatically detects and localizes human faces in an uploaded image.
•	Aging Sign Classification → Classifies detected faces into four categories:
o	Clear Face
o	Dark Spots
o	Puffy Eyes
o	Wrinkles
(with confidence scores for each)
•	Age Prediction → Predicts apparent age (integer).
•	Annotated Image Generation → Creates annotated images with bounding boxes and predictions.
•	Data Export → Download annotated images and a CSV file with detailed predictions.
•	Web Interface → Simple, intuitive UI for uploads and results.
________________________________________
🛠 Tech Stack
Area	Tools / Libraries
Backend   :	Python,Streamlit
Image Ops :	OpenCV, NumPy, Haar Cascades
Model	    : TensorFlow/Keras, ResNet50V2, MobileNetV2
Datasets	: UTKFace (Age), Custom Dataset (Aging Signs)
Frontend	: Streamlit
Evaluation:	Accuracy, Loss, MAE, Confusion Matrix
Exporting	: CSV, Annotated Image
________________________________________
👩‍💻 User Guide
🔎 How It Works
1.	Upload an image through the web interface.
2.	Haar Cascade detects the face location.
3.	The detected face is cropped and preprocessed.
4.	ResNet50V2 → classifies aging signs.
5.	MobileNetV2 → predicts age.
6.	The original image is annotated with predictions.
7.	Results displayed on the results page (with options to download).
🌐 How to Use the Web App
1.	Open the web app in your browser.
2.	Click "Choose an Image" → upload .jpg, .jpeg, or .png.
3.	Use a clear, well-lit face image for best results.
4.	Click "Analyze Image".
5.	View the results: original + annotated image, predictions, confidence scores.
6.	Download annotated image or CSV report.
________________________________________
👨‍💻 Developer Guide
📂 Project Structure
Age-predictor/ 
├── app.py # The main Streamlit application script 
├── requirements.txt # Project dependencies (stays in root)
 ├── README.md # Project documentation (should be in root) 
│├── data/ │ ├── UTKFace/ # Corrected name for the age dataset folder 
                      │ └── custom_dataset/ # Corrected name for the skin dataset folder 
│├── models/ │ ├── age_prediction_model_fast.h5 
                │ ├── dermal_scan_model_best.h5
                              │   └── haarcascade_frontalface_default.xml
│ ├── scripts/ │ ├── main.py # Training script for the skin classifier 
                              │ └── train_age_model.py # Training script for the age model 
│ ├── outputs/ │ └── prediction_log.csv # All generated log files go here
 │ └── test_images/ └── Arpit.jpg # Sample images for testing go here
________________________________________
⚙️ Setup and Installation
1. Prerequisites
•	Python 3.8+
•	pip
2. Clone the Repository
git clone https://github.com/your-username/DermalScan.git
cd DermalScan
3. Create Virtual Environment
# Windows
python -m venv venv
.\venv\Scripts\activate
# macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
A `requirements.txt` file should be created to list all necessary libraries. You can generate it with the following command in your active virtual environment
pip freeze > requirements.txt
Then, install the dependencies:
pip install -r requirements.txt
5. Place Models
Download/pretrained models and place inside models/:
•	age_prediction_model_fast.h5
•	dermal_scan_model_best.h5
•	haarcascade_frontalface_default.xml
________________________________________
▶️ Running the Application
streamlit run app.py________________________________________
📊 Model Information
1. Aging Sign Classifier
•	Model: ResNet50V2 (ImageNet pre-trained, fine-tuned)
•	Task: Multi-class classification
•	Dataset: Custom dataset with 4 categories
•	Performance : 96.7% accuracy
2. Age Predictor
•	Model: MobileNetV2
•	Tasks:
o	Age → regression
•	Dataset: UTKFace
•	Performance Targets:
o	Age: MAE =3.14 years
________________________________________🔬 Replicating Model Training
To replicate the training process for the models used in this project, follow the instructions below.
1. Skin Condition Classifier (ResNet50V2)
•	Training Script: main.py
•	Setup: Download the custom dataset for skin conditions and ensure the DATASET_PATH variable in the script points to its location.
•	Process: The script handles data preprocessing, augmentation, transfer learning from a pre-trained ResNet50V2, and fine-tuning.
2. Age Predictor (MobileNetV2)
•	Training Script: train_age_model.py
•	Setup: Download the UTKFace dataset and ensure the DATASET_PATH variable in the script points to its location.
•	Process: The script handles label parsing from filenames, preprocessing of 128×128 RGB images, and model training for the regression task.
________________________________________
✅ Summary
DermalScan provides an end-to-end ML pipeline for:
•	Detecting facial regions
•	Classifying aging signs
•	Predicting age 
•	Annotating and exporting results
The combination of ResNet50V2 + MobileNetV2  makes DermalScan a robust tool for dermatological and demographic analysis.
