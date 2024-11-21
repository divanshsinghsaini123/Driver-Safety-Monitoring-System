import cv2
import os

# oneDNN custom operations are on.
# You may see slightly different numerical results due to floating-point round-off errors from different computation orders.
# To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import datetime as dt
import numpy as np
import tensorflow as tf









import cv2
import numpy as np

from keras.models import load_model
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from playsound import playsound
from threading import Thread


def start_alarm(sound):
    """Play the alarm sound"""
    playsound('data/alarm.mp3')


classes = ['Closed', 'Open']
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
# INPUT_VIDEO = 'sample/test_1.mp4'
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(INPUT_VIDEO)
model1 = load_model("drowiness_new7.h5")
count = 0
alarm_on = False
alarm_sound = "data/alarm.mp3"
status1 = ''
status2 = ''















print("Tensorflow version:", tf.__version__)

import torch
from keras.models import load_model

print("Script loaded. Import complete")

OBJECT_DETECTION_MODEL_PATH = "models/best.pt"
PREDICTOR_MODEL_PATH = "models/keras_model.h5"
CLASS_NAMES = {0: "No Seatbelt worn", 1: "Seatbelt Worn"}

# Threshold score for the predictor model
THRESHOLD_SCORE = 0.99

SKIP_FRAMES = 1  # skips every 2 frames
MAX_FRAME_RECORD = 500
INPUT_VIDEO = 'sample/test_1.mp4'
OUTPUT_FILE = (
    "output/test_result_"
    + dt.datetime.strftime(dt.datetime.now(), "%Y%m%d%H%M%S")
    + ".mp4"
)

# Colors for drawing bounding boxes
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

# Function to predict the class of a given image
def prediction_func(img):
    # Resize the image
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    # Normalize the image
    img = (img / 127.5) - 1
    # Expand the image
    img = tf.expand_dims(img, axis=0)
    # Predict the class
    pred = predictor.predict(img)
    # Get the index of the class with the highest score
    index = np.argmax(pred)
    # Get the name of the class
    class_name = CLASS_NAMES[index]
    # Get the confidence score of the class
    confidence_score = pred[0][index]
    return class_name, confidence_score

# Load the predictor model
predictor = load_model(PREDICTOR_MODEL_PATH, compile=False)
print("Predictor loaded")

# Ultralytics object detection model : https://docs.ultralytics.com/yolov5/
model = torch.hub.load(
    "ultralytics/yolov5", "custom", path=OBJECT_DETECTION_MODEL_PATH, force_reload=True
)

# Load the video capture
# cap = cv2.VideoCapture(INPUT_VIDEO)
# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Get the size of the video
size = (frame_width, frame_height)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_FILE.rsplit("/", 1)[0], exist_ok=True)
# Create the video writer
# writer = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

print("Analyzing input video...")

# Function to draw a bounding box on an image
def draw_bounding_box(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# Function to draw text on an image
def draw_text(img, x, y, text, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Function to classify the driver in an image
def classify_driver(img):
    y_pred, score = prediction_func(img)
    # If the class is 0 (no seatbelt worn), draw red bounding box
    if y_pred == CLASS_NAMES[0]:
        draw_color = COLOR_RED
    # If the class is 1 (seatbelt worn), draw green bounding box
    elif y_pred == CLASS_NAMES[1]:
        draw_color = COLOR_GREEN
    return y_pred, score, draw_color

# Function to process a frame from the video
def process_frame(frame):
    # Convert the frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Run the object detection model
    results = model(img)
    # Get the bounding boxes
    boxes = results.xyxy[0]
    # Convert the bounding boxes to numpy
    boxes = boxes.cpu()
    # Iterate over the bounding boxes
    for j in boxes:
        # Get the coordinates of the bounding box
        x1, y1, x2, y2, score, y_pred = j.numpy()
        # Convert the coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Crop the image to the bounding box
        img_crop = img[y1:y2, x1:x2]

        # Classify the driver in the cropped image
        y_pred, score, draw_color = classify_driver(img_crop)

        # If the score is above the threshold, draw the bounding box
        if score >= THRESHOLD_SCORE:
            draw_bounding_box(frame, x1, y1, x2, y2, draw_color)
            draw_text(frame, x1 - 10, y1 - 10, f"{y_pred} {str(score)[:4]}", draw_color)

    return frame

# Initialize the frame count
frame_count = 0
# While the video is not finished
while True:
    # Read a frame from the video
    ret, frame = cap.read()




    if not ret:
        break

    height = frame.shape[0]
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
            # print(status1)
            # status1 = classes[pred1.argmax(axis=-1)[0]]
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
            # print(status2)
            # status2 = classes[pred2.argmax(axis=-1)[0]]
            break

        # If the eyes are closed, start counting
        if status1 == 2 and status2 == 2:
            # if pred1 == 2 and pred2 == 2:
            count += 1
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 1)
            # if eyes are closed for 10 consecutive frames, start the alarm
            if count >= 3:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                            2)
                if not alarm_on:
                    alarm_on = True
                    # play the alarm sound in a new thread
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False

    # cv2.imshow("Drowsiness Detector", frame)







    # If the frame is read successfully

    # Increment the frame count
    frame_count += 1

    frame = process_frame(frame)

    # Show the frame
    cv2.imshow("Video feed", frame)

    # If the frame count is above the maximum frame record, break
    if frame_count > MAX_FRAME_RECORD:
        break
    # If the user presses q, break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
# writer.release()

# Destroy all the windows
cv2.destroyAllWindows()

print("Script run complete. Results saved to :", OUTPUT_FILE)
