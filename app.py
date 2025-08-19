import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import seaborn as sns

# Load the trained model
model = load_model("brain_tumor_detection_model.keras")

# Define class labels
class_labels = ['glioma', 'meningioma', 'no tumor', 'pitutary']

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((150, 150))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to make prediction
def make_prediction(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Streamlit UI
st.title("Brain Tumor Detection")

st.write("Upload an image to predict the class of the brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button
if st.button("Predict") and uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = make_prediction(uploaded_file)

    # Display prediction results
    st.subheader("Prediction Results:")
    st.write("Class Probabilities:")
    probs = prediction[0]
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {probs[i]*100:.2f}%")

    # Plot prediction probabilities
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_labels, y=probs)
    plt.xlabel("Tumor Class")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")
    st.pyplot(plt)

    # Display predicted class
    predicted_class = class_labels[np.argmax(prediction)]
    st.subheader(f"Predicted Class: {predicted_class}")
