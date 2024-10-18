import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('model.h5')

# Streamlit app
st.title("Monkeypox Skin Lesion Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Resize to model's input size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    # Display the result
    if predicted_class == 0:
        st.write("Prediction: Monkeypox")
    else:
        st.write("Prediction: Non-Monkeypox")
