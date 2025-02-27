import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Define paths
MODEL_PATH = "imageclassifier.h5"  # Ensure the model is in the same directory as app.py

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

# Custom CSS for better styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #4A90E2;
            text-align: center;
        }
        .sub-title {
            font-size: 20px;
            color: #333;
            text-align: center;
        }
        .uploaded-image {
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            font-size: 24px;
            text-align: center;
            color: white;
            font-weight: bold;
            margin-top: 20px;
        }
        .happy { background-color: #4CAF50; }  /* Green for Happy */
        .sad { background-color: #E74C3C; }  /* Red for Sad */
        .confidence-bar {
            margin-top: 20px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="main-title">😊 Happy or 😢 Sad Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to classify its emotion.</p>', unsafe_allow_html=True)

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
        st.image(image, caption="Uploaded Image", use_column_width=True, clamp=True)

        # Prediction Button
        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing... 🔄"):
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
                color_class = "happy" if predicted_class == "Happy" else "sad"
                st.markdown(
                    f'<div class="prediction-box {color_class}">Prediction: {predicted_class}</div>',
                    unsafe_allow_html=True
                )

                # Confidence Score Visualization
                confidence_score = confidence if predicted_class == "Happy" else 1 - confidence
                st.markdown(f'<div class="confidence-bar">Confidence: {confidence_score * 100:.2f}%</div>', unsafe_allow_html=True)
                st.progress(float(confidence_score))  # Convert to native Python float

                # Handle edge cases (e.g., confidence close to threshold)
                if abs(confidence - confidence_threshold) < 0.1:
                    st.warning("The prediction is close to the threshold. Consider adjusting the threshold for better results.")

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