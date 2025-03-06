import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import requests
from PIL import Image
import os

# Hugging Face model URL
MODEL_URL = "https://huggingface.co/ddv2311/imageclassifier/resolve/main/imageclassifier.h5"
MODEL_PATH = "imageclassifier.h5"

# Function to download model from Hugging Face
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... ⏳")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully! ✅")

# Load the trained model
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please ensure the model file exists.")
            return None
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define class labels
class_names = ["Sad", "Happy"]  # Adjust based on your model's training labels

# Function to preprocess image
def preprocess_image(image):
    try:
        image = np.array(image.convert("RGB"))  # Ensure image is in RGB format
        image = cv2.resize(image, (256, 256))  # Resize to match model input shape
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit UI Styling
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")

# Title
st.markdown('<h1 style="text-align:center; color:#4A90E2;">😊 Happy or 😢 Sad Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Upload an image to classify its emotion.</p>', unsafe_allow_html=True)

# Sidebar for confidence threshold
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,  # Default threshold
        step=0.01,
        help="Adjust the threshold for classification. Higher values make the model more conservative."
    )

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prediction Button
        if st.button("Analyze Emotion"):
            with st.spinner("Downloading & analyzing... 🔄"):
                # Download model if not present
                download_model()

                # Load model
                model = load_model(MODEL_PATH)
                if model is None:
                    st.error("Model could not be loaded. Please check the model file.")
                    st.stop()

                # Preprocess image
                processed_image = preprocess_image(image)
                if processed_image is None:
                    st.error("Image could not be processed. Please upload a valid image.")
                    st.stop()

                # Make prediction
                prediction = model.predict(processed_image)
                confidence = prediction[0][0]  # Assuming binary classification with sigmoid output
                predicted_class = class_names[int(confidence >= confidence_threshold)]  # Use adjustable threshold

                # Display Result
                color_class = "✅ Happy" if predicted_class == "Happy" else "❌ Sad"
                st.markdown(
                    f'<h3 style="text-align:center; color:#4CAF50;">Prediction: {predicted_class}</h3>'
                    if predicted_class == "Happy"
                    else f'<h3 style="text-align:center; color:#E74C3C;">Prediction: {predicted_class}</h3>',
                    unsafe_allow_html=True
                )

                # Confidence Score
                confidence_score = confidence if predicted_class == "Happy" else 1 - confidence
                st.markdown(f'<p style="text-align:center;">Confidence: {confidence_score * 100:.2f}%</p>', unsafe_allow_html=True)
                st.progress(float(confidence_score))  # Convert to native Python float

        # Clear Button
        if st.button("Clear Image"):
            uploaded_file = None
            st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    '<p style="text-align:center; margin-top:20px;">Made with ❤️ using Streamlit & TensorFlow!</p>',
    unsafe_allow_html=True
)
