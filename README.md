# emotion_detection_app
AI-based real-time facial emotion detection system for mental health monitoring using CNN and webcam.

# AI Based Emotion Detection System 🧠

## Mental Health Monitoring using Webcam

A real-time emotion detection web application that uses 
a Convolutional Neural Network (CNN) trained on the 
FER2013 dataset to detect facial emotions through webcam 
and analyze mental health indicators.

## Features
- 🎥 Real-time webcam emotion detection
- 🧠 7 emotion classes: Angry, Disgust, Fear, 
  Happy, Neutral, Sad, Surprise
- 💊 Depression risk analysis
- 😓 Stress level monitoring  
- 💬 Therapy assistance and coping tips
- 🌐 Interactive web interface using Streamlit

## Dataset
- FER2013 (Facial Expression Recognition 2013)
- Source: Kaggle
- 28,709 training images
- 7,178 testing images

## Model
- Architecture: Custom CNN (3 Conv layers)
- Training Accuracy: 66.56%
- Framework: PyTorch

## Tech Stack
- Python 3.14
- PyTorch
- OpenCV
- Streamlit
- Torchvision

## How to Run
1. Clone the repository
2. Install dependencies
   pip install -r requirements.txt
3. Run the app
   python -m streamlit run app.py

## Project Structure
emotion_detection/
    app.py
    emotion_model.pth
    emotion_detection_m3.ipynb
    archive/
        train/
        test/
