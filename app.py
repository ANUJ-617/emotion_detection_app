import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import urllib.request
import os

# CNN Model definition
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Settings
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_status(emotion):
    if emotion in ['Sad', 'Fear', 'Disgust']:
        return "High Risk", "High Stress", [
            "Please consider talking to a mental health professional.",
            "Try journaling your thoughts and feelings daily.",
            "Reach out to a trusted friend or family member.",
            "Practice self-compassion — it's okay to not be okay."
        ]
    elif emotion in ['Angry']:
        return "Moderate Risk", "High Stress", [
            "Try the 4-7-8 breathing technique to calm down.",
            "Take a 10 minute break away from screens.",
            "Break your tasks into smaller manageable steps."
        ]
    elif emotion in ['Happy']:
        return "Low Risk", "Low Stress", [
            "Great mood! Keep doing what makes you happy!",
            "Share your positivity with someone today!"
        ]
    elif emotion in ['Surprise']:
        return "Low Risk", "Moderate Stress", [
            "Take a deep breath and process your thoughts.",
            "Write down what surprised you today."
        ]
    else:
        return "Low Risk", "Low Stress", [
            "You seem calm and stable. Keep it up!",
            "Try something creative today to boost your mood!"
        ]

def detect_emotion_from_image(image, model, device):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return image, "No face detected"

        for (x, y, w, h) in faces:
            face = image.crop((x, y, x+w, y+h))
            input_tensor = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                emotion = emotion_labels[predicted.item()]

            img_array = cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_array, emotion, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return Image.fromarray(img_array), emotion
    except Exception as e:
        return image, "Error detecting emotion"

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
    model.eval()
    return model, device

# Streamlit UI
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("🧠 AI Based Emotion Detection System")
st.subheader("Mental Health Monitoring")

model, device = load_model()

# Mode selection
mode = st.radio("Select Mode:", ["📷 Upload Photo", "🎥 Live Webcam (Local only)"])

if mode == "📷 Upload Photo":
    st.markdown("### Upload a photo to detect emotion")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Your Photo")
            result_image, emotion = detect_emotion_from_image(image, model, device)
            st.image(result_image, use_column_width=True)

        with col2:
            st.markdown("### Analysis Results")
            if emotion != "No face detected" and emotion != "Error detecting emotion":
                depression, stress, tips = get_status(emotion)
                st.metric("Detected Emotion", emotion)
                st.metric("Depression Risk", depression)
                st.metric("Stress Level", stress)
                st.markdown("**Therapy Tips:**")
                for i, tip in enumerate(tips, 1):
                    st.markdown(f"{i}. {tip}")
            else:
                st.warning(emotion)

elif mode == "🎥 Live Webcam (Local only)":
    st.warning("⚠️ Webcam mode works only when running locally!")
    st.code("python -m streamlit run app.py", language="bash")
    
    run = st.checkbox("Start Webcam")
    frame_placeholder = st.empty()
    col1, col2 = st.columns(2)

    if run:
        cap = cv2.VideoCapture(0)
        with col2:
            emotion_placeholder = st.empty()
            depression_placeholder = st.empty()
            stress_placeholder = st.empty()
            tips_placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            emotion = "No face detected"

            for (x, y, w, h) in faces:
                face = Image.fromarray(frame[y:y+h, x:x+w])
                input_tensor = transform(face).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    emotion = emotion_labels[predicted.item()]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            if emotion != "No face detected":
                depression, stress, tips = get_status(emotion)
                with col2:
                    st.metric("Detected Emotion", emotion)
                    st.metric("Depression Risk", depression)
                    st.metric("Stress Level", stress)
                    st.markdown("**Therapy Tips:**")
                    for i, tip in enumerate(tips, 1):
                        st.markdown(f"{i}. {tip}")
        cap.release()
