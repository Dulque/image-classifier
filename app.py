# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("🐶🐱 Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    pred = model.predict(preprocess_image(img))[0][0]
    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    st.subheader(f"Prediction: {label} ({pred:.2f})")
