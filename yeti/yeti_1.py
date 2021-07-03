import cv2
import numpy as np
import sys
import logging as log
import datetime as dt
from time import sleep

#cascPath = cv2.data.haarcascades + "haarcascade_eye.xml"
#cascPath = "/Users/martin/Google Drive/Aktenkoffer/Packages/YET/yeti/haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier("./models/haarcascade_eye.xml")
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(1)
detected = False

white_lower = np.array([200, 200, 200], dtype = "uint8")
white_upper = np.array([255, 255, 255], dtype = "uint8")

if not video_capture.isOpened():
        print('Unable to load camera.')
        exit()

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    cv2.imshow('YETI', frame)
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200)
    )

    if len(eyes) > 0:
        eye = eyes[0]
        print("Eye detected")
        detected = True
    else:
        detected = False

    if detected:
        (x, y, w, h) = eye
        #ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        eye_frame = gray[y:y+h,x:x+w]
        half_w = int(w/2)
        left_frame = eye_frame[y:y+h, x:x+half_w]
        right_frame = eye_frame[y:y+h, half_w:x]
        cv2.imshow('YETI-eye', eye_frame)
        #cv2.imshow('YETI-right', right_frame)

    cv2.imshow('YETI', gray)
        
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


def augment_frame(left_frame):
    cv2.putText(left_frame, #numpy array on which text is written
    "left", #text
    (50, 50), #position at which writing has to start
    cv2.FONT_HERSHEY_SIMPLEX, #font family
    50, #font size
    (209, 80, 0, 255), #font color
    3) #font stroke
