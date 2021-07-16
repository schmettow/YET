# LIBRARIES

import cv2
import numpy as np
import sys
import logging as log
import datetime as dt
from time import sleep

# PREPARATIONS

#cascPath = cv2.data.haarcascades + "haarcascade_eye.xml"
#cascPath = "/Users/martin/Google Drive/Aktenkoffer/Packages/YET/yeti/haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier("./models/haarcascade_eye.xml")
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(1)
detected = False

if video_capture.isOpened():
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps =  video_capture.get(cv2.CAP_PROP_FPS)
    dim = (width, height)
else:
    print('Unable to load camera.')
    exit()

Detected = False

# FAST LOOP
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # FRAME PROCESSING
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if not Detected:
        eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200))


    # AUTOMATIC TRANSITIONAL
    if len(eyes) > 0:
        eye = eyes[0]
        print("Eye detected")
        Detected = True
        center_ratio = .85
    # once eye is detected, yeti stays there

    # CONDITIONAL PROCESSING
    if Detected:
        (x, y, w, h) = eye
        eye_frame = cv2.resize(frame[y:y+h,x:x+w], dim, interpolation = cv2.INTER_AREA)
        left_frame = eye_frame[0:height, 0:int(width/2)]
        right_frame = eye_frame[0:height, int(width/2):width]
        left_sum = np.sum(left_frame)
        right_sum = np.sum(right_frame)
        LR_ratio = left_sum/right_sum
        Position = "center"
        if LR_ratio < center_ratio - .05:
            Position = "right"
        if LR_ratio > center_ratio + .05:
            Position = "left"
            
        
        
    # PRESENTITIONALS
    
    if Detected:
        if Position == "center":
            font_col = (255,0,0)
        elif Position == "left":
            font_col = (0,0,255)
        elif Position == "right":
            font_col = (0,255,0)
        
        out_frame = cv2.putText(eye_frame, f"{LR_ratio}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
        out_frame = cv2.putText(eye_frame, f"{Position}", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2,
        font_col, thickness = 8)
    else:
        out_frame = frame
    
    #out_frame = cv2.resize(out_frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('YETI_2', out_frame)
    
    # INTERACTIVE TRANSITIONALS
    
    # This is almost cryptic
    # Can we use the PyGame event handler, instead?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & Detected & 0xFF == ord('c'):
        center_ratio = LR_ratio
    

    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

