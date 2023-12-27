import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from PIL import Image
import pickle

model_version = 1
# loaded_model = tf.keras.models.load_model(f"/models/model_{model_version}")
#loading the saved model 
loaded_model = pickle.load(open(f"trained_model.sav", 'rb'))


project_info = """
This is a Potato disease classification project. 
The model predicts the following classes: \n
['Potato Early blight', 'Potato Late blight', 'Potato healthy']
"""
dataset_link='https://www.kaggle.com/datasets/arjuntejaswi/plant-village'

# Display Kaggle link and project information
st.markdown(f"[Link to the Dataset on Kaggle]({dataset_link})", unsafe_allow_html=True)
st.write(project_info)


uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = loaded_model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence

def display_suggestions(predicted_class):
    if predicted_class == 'Potato___Early_blight':
        st.subheader("Suggestions for Potato Early Blight:")
        st.write("- Apply fungicides recommended for early blight control.")
        st.write("- Remove infected leaves to prevent the spread of the disease.")
    elif predicted_class == 'Potato___Late_blight':
        st.subheader("Suggestions for Potato Late Blight:")
        st.write("- Use fungicides suitable for late blight management.")
        st.write("- Ensure proper spacing between plants for better air circulation.")
    elif predicted_class == 'Potato___healthy':
        st.subheader("Suggestions for Healthy Potato Plant:")
        st.write("- Continue with regular care and monitoring.")
        st.write("- Implement crop rotation to prevent soil-borne diseases.")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.resize((256, 256))
    st.image(image, caption="Uploaded Image", use_column_width=False, width=300)

    predicted_class, confidence = predict(image)
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
    # Display suggestions based on predicted class
    display_suggestions(predicted_class)

