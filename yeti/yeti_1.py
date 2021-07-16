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


# FAST LOOP
while True:

    # Capture frame-by-frame
    # Assumingly, read() waits for the next frame. 
    # A side effect is that the key events are also only sampled at that rate.
    # Which can feel sluggish with low frame rates.
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # FRAME PROCESSING
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(250, 250)
    )


    # AUTOMATIC TRANSITIONALS
    if len(eyes) > 0:
        eye = eyes[0]
        print("Eye detected")
        Detected = True ## write status variable in capital
    else:
        Detected = False
    
    # CONDITIONAL PROCESSING
    if Detected:
        (x, y, w, h) = eye
        eye_frame = cv2.resize(frame[y:y+h,x:x+w], dim, interpolation = cv2.INTER_AREA)
        
        
    # PRESENTITIONALS
    if Detected:
        out_frame = cv2.putText(eye_frame, f"Hello Eye!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness = 8)
    else:
        out_frame = frame
    
    cv2.imshow('YETI_1', out_frame)
    
    
    # INTERACTIVE TRANSITIONALS
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


