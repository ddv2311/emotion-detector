# **Emotion Detection App** 😊😢

Welcome to the **Emotion Detection App**! This app uses a deep learning model to classify images as either **Happy** or **Sad**. Built with **Streamlit** and **TensorFlow**, it provides a user-friendly interface for uploading images and viewing predictions.

---

## **Features** ✨

- **Image Classification**: Upload an image, and the app will classify it as "Happy" or "Sad."
- **Confidence Threshold**: Adjust the confidence threshold using a slider to control the sensitivity of predictions.
- **Visual Feedback**: 
  - Color-coded prediction boxes (Green for "Happy," Red for "Sad").
  - Confidence score displayed as a percentage and progress bar.
- **Edge-Case Handling**: Warns users when predictions are close to the threshold.
- **Responsive Design**: Clean and modern UI with custom styling.

---

## **Installation** 🛠️

Follow these steps to set up and run the app locally.

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)

### **Steps**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ddv2311/emotion-detector.git
   cd emotion-detection-app```
   
2.  **Create virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. **Download the model**:
   
     Run .ipynb file to get model
   
5. **Run the app**:
   ```bash
   streamlit run app.py
   ```
     

