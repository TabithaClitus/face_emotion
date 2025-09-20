😃 Face Emotion Detector

APP DEMO:https://faceemotion-vovdmxan8wfh6flhqmdwvk.streamlit.app/

A real-time Face Emotion Detection system using Convolutional Neural Networks (CNN) that identifies emotions from facial expressions. Detects emotions such as Happy, Sad, Angry, Surprise, Neutral, Fear, and Disgust in images or live webcam feed.

🎯 Features

Real-time emotion detection from webcam.

Works with images and live video.

Supports 7 common emotions:

😄 Happy | 😢 Sad | 😠 Angry | 😲 Surprise | 😐 Neutral | 😨 Fear | 🤢 Disgust

Built with TensorFlow/Keras and OpenCV.

Easy to train on your own dataset.

📸 Demo

<img width="1479" height="494" alt="Screenshot 2025-09-20 085411" src="https://github.com/user-attachments/assets/4677560d-769e-4aab-8230-1737e7d25d85" />

<img width="1688" height="869" alt="Screenshot 2025-09-20 085450" src="https://github.com/user-attachments/assets/af38a417-4784-4195-a1c7-ef2434cb61a4" />

<img width="1492" height="829" alt="Screenshot 2025-09-20 085459" src="https://github.com/user-attachments/assets/344c086b-ad72-438f-95ff-1d5c26d63729" />

<img width="1306" height="647" alt="Screenshot 2025-09-20 085542" src="https://github.com/user-attachments/assets/db38180a-de0c-4ab4-9e57-9ac2adb6ff0f" />

📂 Dataset

Uses the FER-2013 dataset, a large collection of 48x48 grayscale images of facial expressions labeled with 7 emotions.

Download FER-2013 Dataset from Kaggle

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/face-emotion-detector.gitcd face-emotion-detector


Install dependencies:

pip install -r requirements.txt

requirements.txt:

tensorflow

keras

numpy

matplotlib

🚀 Usage

1️⃣ Training the Model

python train_model.py


Trains the CNN on FER-2013 and saves the trained model as emotion_model.h5.

2️⃣ Real-Time Emotion Detection

python detect_emotion.py

🏗️ Model Architecture

Input: 48x48 grayscale image

Conv2D Layers: 32, 64, 128 filters with ReLU activation

MaxPooling2D & Dropout: for regularization

Dense Layers: Fully connected layers with ReLU

Output Layer: 7 neurons with Softmax activation

✅ Conclusion

This project demonstrates how deep learning and computer vision can be combined to recognize human emotions from facial expressions in real time.
With applications in human-computer interaction, healthcare, education, customer service, and entertainment, emotion detection has the potential to make systems more intelligent and empathetic.
