import streamlit as st

import os
st.write('''
# Derma Scan
## the ultimate ageing symptom detector
''')

import streamlit as st

from image_ops import loader, preprocess, label

from models import predict_age, predict_feature, model_loader


@st.cache_resource
def cascade_loader():
    cascade=loader.load_cascade()
    return cascade
    
@st.cache_data
def age_model_loader():
    age_model =model_loader.load_keras_model(r'D:\Projects\skin-age-detection\models\age_mobilenet_regression_stratified_old.h5')
    return age_model
@st.cache_data
def feature_model_loader():
    feature_model= model_loader.load_keras_model(r'D:\Projects\skin-age-detection\models\mobilenet_effnet_head.h5')
    return feature_model



with st.spinner('loading our best models for you'):
    cascade=cascade_loader()
    age_model=age_model_loader()
    feature_model=feature_model_loader()
st.write("Models loaded successfully!")

st.markdown('### Upload your image here ensure proper lighting and no wearables(earrings, glasses, etc) for more accurate predictions!')
file = st.file_uploader('upload the image file: supported formats png jpeg jpg')

with st.spinner('processing your image'):
    if not file:
        st.write("ERROR: Test image not found")
    else:
        image_bytes = file.getvalue()
        processed_face = preprocess.preprocess_image(image_bytes)
    if processed_face is not None:
        label.labeled_image

