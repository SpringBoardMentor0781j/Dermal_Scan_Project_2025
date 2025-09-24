# Save this file as: tmp.py

import streamlit as st                                              # The main library for building the web app.
import numpy as np                                                  # Required for handling image data in a numerical format.
import cv2                                                          # Used to decode the uploaded image file into a format we can work with.
from PIL import Image                                               # A friendly library for handling image objects.
from models.predict_feature import load_model, predict_features     # Correctly imports our PURE functions from the 'models' package.

# --- Page Configuration ---
st.set_page_config(                                                 # This function sets up the basic properties of our web page.
    page_title="Facial Feature Analyzer",                           # This is the title that will appear in the browser tab.
    page_icon="🔬",                                                # This is the little icon (favicon) in the browser tab.
    layout="centered"                                               # This keeps our app looking neat and tidy in the center of the page.
)

# --- Model Caching (Handled entirely within the Streamlit App) ---
MODEL_PATH = "models/efficientnet_b0_face_classifier_finetuned.h5"  # This path points to the model file inside the 'models' directory.

@st.cache_resource # This Streamlit decorator caches the output of this function.
def get_model(path):
    """
    This is a wrapper function that calls the real model loader from our module
    and caches the result. This keeps all Streamlit-specific code in this file.
    """
    # Call the actual model loading function from our pure module.
    return load_model(path)

# Load our trained model using the cached wrapper function.
model = get_model(MODEL_PATH)                                       # This line calls our new cached function to get the model.

# --- User Interface Elements ---
st.title("🔬 Facial Feature Percentage Analyzer")                   # This displays the main title at the top of the web page.

st.write(                                                           # This displays a paragraph of text explaining what the app does.
    "Upload a photo of a face, and this tool will analyze it to predict the likelihood of various skin features."
)

# This creates the file uploader widget on the page.
uploaded_file = st.file_uploader(                                   # This Streamlit function creates a drag-and-drop or browse-to-upload box.
    "Choose an image to analyze...",                                # This is the instructional text shown inside the uploader.
    type=["jpg", "jpeg", "png"]                                     # This ensures users can only upload files with these common image extensions.
)

# This block of code only runs if the user has successfully uploaded a file.
if uploaded_file is not None:
    # Convert the uploaded file into a format that OpenCV can understand.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8) # We read the file's raw data (bytes) and turn it into a NumPy array.
    opencv_image = cv2.imdecode(file_bytes, 1)                      # OpenCV then decodes this array into a proper image format (BGR color order).
    
    # For displaying the image correctly, convert it from BGR (OpenCV) to RGB.
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)# This line swaps the blue and red channels.

    # Display the uploaded image on our web page.
    st.image(opencv_image_rgb, caption='Your Uploaded Image', use_column_width=True) # This shows the image to the user with a caption.

    # --- Prediction and Results Display ---
    st.write("### 🧠 Analyzing...")                                  # We add a sub-header to let the user know what's happening.
    
    with st.spinner('The model is having a look...'):               # This creates a loading spinner while the model is working.
        # Call our prediction function, passing it the loaded model and the user's image.
        prediction_results = predict_features(model, opencv_image_rgb) # The function returns a dictionary of feature percentages.

    st.write("### 📊 Analysis Results")                             # After prediction is done, show a new sub-header for the results.
    
    # Loop through each result (feature and its predicted percentage) in the dictionary.
    for feature, percentage in prediction_results.items():
        # Format the feature name to be more readable (e.g., 'clear_face' becomes 'Clear Face').
        formatted_name = feature.replace('_', ' ').title()
        # Display each feature and its corresponding percentage.
        st.write(f"**{formatted_name}**: {percentage:.2%}")          # Formats the number as a percentage with two decimal places.