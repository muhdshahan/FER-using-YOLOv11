import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import Counter

# Setup
RUN_NAME = 'facial_emotion_fer2013_v12'
model_path = os.path.join('runs', 'classify', RUN_NAME, 'weights', 'best.pt')
emotion_model = YOLO(model_path)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_classes = emotion_model.names
font = cv2.FONT_HERSHEY_SIMPLEX

st.set_page_config(page_title="Real-Time Emotion Recognition", page_icon="😊", layout="centered")
st.title("Real-Time Facial Emotion Recognition")
st.markdown("Detect your **facial emotion** live using your webcam powered by YOLOv11 and Streamlit.")

# Session State Setup
if "capturing" not in st.session_state:
    st.session_state.capturing = False
if "emotions" not in st.session_state:
    st.session_state.emotions = []

start_cam = st.button("Start Camera")
stop_cam = st.button("Stop Camera")

if start_cam:
    st.session_state.capturing = True
    st.session_state.emotions = [] 

if stop_cam:
    st.session_state.capturing = False

# Camera Feed
FRAME_WINDOW = st.image([])

if st.session_state.capturing:
    cap = cv2.VideoCapture(0)
    st.info("Camera is active — press **Stop Camera** to end detection.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frame detected. Please check your webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (48, 48))
            input_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2BGR)

            results = emotion_model.predict(input_face, verbose=False)
            probs = results[0].probs
            predicted_class_id = probs.top1
            predicted_emotion = emotion_classes[predicted_class_id]
            print(predicted_class_id, emotion_classes)
            st.session_state.emotions.append(predicted_emotion)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), font, 0.8, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Streamlit needs break condition
        if not st.session_state.capturing:
            break

    cap.release()
    cv2.destroyAllWindows()

# Display Final Emotion 
if not st.session_state.capturing and st.session_state.emotions:
    st.subheader("Final Emotion Summary")
    emotion_counts = Counter(st.session_state.emotions)

    # If there are emotions other than neutral, remove neutral from consideration
    if len(emotion_counts) > 1 and "neutral" in emotion_counts:
        del emotion_counts["neutral"]
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    st.write(f"Your facial expression was: **{most_common_emotion.upper()}**")
