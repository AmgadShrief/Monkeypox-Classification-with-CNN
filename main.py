import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import urllib.request
import tempfile

# URL to the model file on Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1rDI2QTo7jyHiw4fX5Ls5TSaBJSwYzwwX"

# Use a temporary directory to store the downloaded model
temp_dir = tempfile.gettempdir()
MODEL_PATH = os.path.join(temp_dir, "downloaded_model.h5")

# Download the model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... This may take a while!"):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Download completed!")

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Streamlit app code
st.title("Monkeypox Skin Lesion Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Resize to model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    try:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])

        # Display the result
        if predicted_class == 0:
            st.write("Prediction: Monkeypox")
        else:
            st.write("Prediction: Non-Monkeypox")
    except Exception as e:
        st.write(f"Error making prediction: {e}")
        
