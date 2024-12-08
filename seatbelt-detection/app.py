import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from playsound import playsound
from threading import Thread
import torch
import streamlit as st
import tempfile


# Function to play alarm sound
def start_alarm(sound):
    """Play the alarm sound"""
    playsound('data/alarm.mp3')


# Constants
classes = ['Closed', 'Open']
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
MODEL_PATH = "drowiness_new7.h5"
INPUT_VIDEO = None
frame_count = 0
alarm_on = False
count = 0

# Load the drowsiness detection model
model1 = load_model(MODEL_PATH)

# Load the object detection model (YOLO)
OBJECT_DETECTION_MODEL_PATH = "models/best.pt"
PREDICTOR_MODEL_PATH = "models/keras_model.h5"
CLASS_NAMES = {0: "No Seatbelt worn", 1: "Seatbelt Worn"}
THRESHOLD_SCORE = 0.99

predictor = load_model(PREDICTOR_MODEL_PATH, compile=False)
model = torch.hub.load("ultralytics/yolov5", "custom", path=OBJECT_DETECTION_MODEL_PATH, force_reload=True)


# Function to predict the class of a given image for seatbelt detection
def prediction_func(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = (img / 127.5) - 1
    img = tf.expand_dims(img, axis=0)
    pred = predictor.predict(img)
    index = np.argmax(pred)
    class_name = CLASS_NAMES[index]
    confidence_score = pred[0][index]
    return class_name, confidence_score


# Function to draw bounding boxes
def draw_bounding_box(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


# Function to draw text on an image
def draw_text(img, x, y, text, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


# Function to classify the driver based on seatbelt detection
def classify_driver(img):
    y_pred, score = prediction_func(img)
    if y_pred == CLASS_NAMES[0]:
        draw_color = (255, 0, 0)  # Red for no seatbelt
    else:
        draw_color = (0, 255, 0)  # Green for seatbelt worn
    return y_pred, score, draw_color


# Function to process each frame for both drowsiness and seatbelt detection
def process_frame(frame):
    global count, alarm_on
    height = frame.shape[0]

    # Drowsiness detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)

        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1 + h1, x1:x1 + w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model1.predict(eye1)
            status1 = np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model1.predict(eye2)
            status2 = np.argmax(pred2)
            break

        if status1 == 2 and status2 == 2:  # Eyes closed
            count += 1
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 1)
            if count >= 3:  # Alarm on after 3 frames
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                            2)
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=start_alarm, args=("data/alarm.mp3",))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False

    # Seatbelt detection using YOLO model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    boxes = results.xyxy[0]  # Bounding boxes for detected objects

    for j in boxes:
        x1, y1, x2, y2, score, class_id = j.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name, confidence = prediction_func(frame[y1:y2, x1:x2])

        if confidence >= THRESHOLD_SCORE:
            draw_bounding_box(frame, x1, y1, x2, y2, (0, 255, 0))  # Green for seatbelt worn
            draw_text(frame, x1, y1, f"{class_name} {confidence:.2f}", (0, 255, 0))

    return frame


# Streamlit UI setup
st.title("Driver Monitoring System")
st.write("This system detects drowsiness and seatbelt usage in real-time.")

# Upload video using Streamlit
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])

if uploaded_video:
    # Temporarily save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_video.read())
        temp_video_path = temp_file.name

    # Read video file
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

st.write("Script completed.")
