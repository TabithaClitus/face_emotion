import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os



# Try to load model safely
try:
    st.write("⏳ Loading model...")
    model = load_model("emotion_model_final.h5")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()  # stop execution if model can't load

# Emotion labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("😊 Emotion Detector")
st.write("Upload a face image and the model will predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        img = Image.open(uploaded_file).convert("L")  # grayscale
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        st.write("🔄 Preprocessing image...")
        img = img.resize((48, 48))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

        # Predict
        st.write("🤖 Running prediction...")
        pred = model.predict(img_array)[0]
        predicted_class = class_labels[np.argmax(pred)]

        st.write(f"### 🎯 Predicted Emotion: **{predicted_class}**")

        # Probability chart
        st.bar_chart(dict(zip(class_labels, pred)))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
