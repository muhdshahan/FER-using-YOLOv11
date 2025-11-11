import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import Counter
from PIL import Image
import platform

# Setup
RUN_NAME = 'facial_emotion_fer2013_v12'
model_path = os.path.join('runs', 'classify', RUN_NAME, 'weights', 'best.pt')
emotion_model = YOLO(model_path)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_classes = emotion_model.names
font = cv2.FONT_HERSHEY_SIMPLEX

# Streamlit page config
st.set_page_config(page_title="Real-Time Emotion Recognition", page_icon="😊", layout="centered")
st.title("Real-Time Facial Emotion Recognition")
st.markdown("Detect your **facial emotion** live using your webcam or uploaded image — powered by YOLOv11 and Streamlit.")

# Detect environment
is_local = "windows" in platform.system().lower() or "darwin" in platform.system().lower()

# Session state setup
if "capturing" not in st.session_state:
    st.session_state.capturing = False
if "emotions" not in st.session_state:
    st.session_state.emotions = []

# Buttons
if is_local:
    start_cam = st.button("🎥 Start Camera")
    stop_cam = st.button("🛑 Stop Camera")

    if start_cam:
        st.session_state.capturing = True
        st.session_state.emotions = []

    if stop_cam:
        st.session_state.capturing = False

FRAME_WINDOW = st.image([])


# LOCAL MODE (webcam)
if is_local and st.session_state.capturing:
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
            st.session_state.emotions.append(predicted_emotion)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), font, 0.8, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not st.session_state.capturing:
            break

    cap.release()


# CLOUD MODE (image upload)
if not is_local:
    st.info("🖼 Running on Streamlit Cloud — upload an image for emotion detection.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            st.warning("No face detected. Try another image.")
        else:
            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (48, 48))
                input_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2BGR)

                results = emotion_model.predict(input_face, verbose=False)
                probs = results[0].probs
                predicted_class_id = probs.top1
                predicted_emotion = emotion_classes[predicted_class_id]

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, predicted_emotion, (x, y - 10), font, 0.8, (0, 255, 0), 2)

                st.session_state.emotions.append(predicted_emotion)

            st.image(image, caption="Detected Emotion", use_container_width=True)


# Display Final Emotion
if not st.session_state.capturing and st.session_state.emotions:
    st.subheader("Final Emotion Summary")
    emotion_counts = Counter(st.session_state.emotions)

    # Remove neutral if other emotions present
    if len(emotion_counts) > 1 and "neutral" in emotion_counts:
        del emotion_counts["neutral"]

    most_common_emotion = emotion_counts.most_common(1)[0][0]
    st.success(f"Your facial expression was: **{most_common_emotion.upper()}** 😄")
