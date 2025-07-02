import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# ------------------ Streamlit Page Config ------------------ #
st.set_page_config(
    page_title="Deepfake Image Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

# ------------------ Model Download & Load ------------------ #
@st.cache_resource
def load_model():
    model_path = "final_model.h5"
    file_id = "1AXnbi7GVCho8oOTkV2y5bWcVreBH2rFG"  # 🔁 Replace with your actual file ID
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

model = load_model()

# ------------------ Image Preprocessing ------------------ #
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------ UI Styling ------------------ #
st.markdown(
    """
    <style>
    /* Light & Dark mode UI styling */
    .stApp {
        background-color: #f5f7fa;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 2rem 1rem;
    }
    .css-1d391kg h1 {
        font-weight: 800;
        font-size: 2.8rem;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }
    .stFileUploader > div {
        border: 2px dashed #3b82f6;
        border-radius: 12px;
        padding: 1.8rem;
        background-color: white;
        transition: border-color 0.3s ease;
    }
    .stFileUploader > div:hover {
        border-color: #2563eb;
    }
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        max-width: 100%;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .prediction-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6 !important;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }
    .stInfo {
        background-color: #e0f2fe;
        border-left: 5px solid #3b82f6;
        color: #0369a1;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0f172a;
            color: #e0e7ff;
        }
        .css-1d391kg h1 {
            color: #e0e7ff;
        }
        .stFileUploader > div {
            background-color: #1e293b;
        }
        .prediction-text {
            color: #93c5fd;
        }
        .confidence-text {
            color: #bfdbfe;
        }
        .stProgress > div > div > div > div {
            background-color: #60a5fa !important;
        }
        .stButton>button {
            background-color: #3b82f6;
        }
        .stButton>button:hover {
            background-color: #60a5fa;
        }
        .stInfo {
            background-color: #1e293b;
            border-left: 5px solid #3b82f6;
            color: #93c5fd;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Main App ------------------ #
st.title("🕵️‍♂️ Deepfake Image Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    label = "Real" if prediction >= 0.5 else "Fake"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    color_map = {
        "Real": "#10b981",  # green
        "Fake": "#ef4444"   # red
    }

    st.markdown(
        f"<div class='prediction-text'>Prediction: <span style='color: {color_map[label]};'>{label}</span></div>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='confidence-text'>Confidence: {confidence:.2%}</div>", 
        unsafe_allow_html=True
    )
    st.progress(int(confidence * 100))
else:
    st.info("Please upload an image to get a prediction.")
