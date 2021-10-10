## Yeti2: A simple proof-of-concept for using split-frame cornea brightness

YETI = 2
YETI_NAME = "Yeti" + str(YETI)
TITLE = "split-frame cornea brightness eye tracking"
AUTHOR = "M Schmettow"


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
eyeCascade = cv2.CascadeClassifier("./trained_models/haarcascade_eye.xml")
log.basicConfig(filename='Yeti_2.log',level=log.INFO)

YET = cv2.VideoCapture(1)

if YET.isOpened():
    width = int(YET.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(YET.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps =  YET.get(cv2.CAP_PROP_FPS)
    dim = (width, height)
else:
    print('Unable to load camera.')
    exit()

## INITIAL STATE
Detected = False ## Eye mot detected.
Eyes = []

# FAST LOOP
while True:

    # FRAME PROCESSING
    ret, Frame = YET.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    F_gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    
    # AUTOMATIC TRANSITIONAL
    if len(Eyes) > 0:
        eye = Eyes[0]
        print("Eye detected")
        Detected = True
    # once eye is detected, yeti stays in the frame

    # CONDITIONAL FRAME PROCESSING
    center_ratio = 0.8

    if not Detected:
        Eyes = eyeCascade.detectMultiScale(
            F_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200))
    
    if Detected:
        (x, y, w, h) = eye
        F_eye = cv2.resize(Frame[y:y+h,x:x+w], dim, interpolation = cv2.INTER_AREA)
        F_left =  F_eye[0:height, 0:int(width/2)]
        F_right = F_eye[0:height, int(width/2):width]
        bright_left = np.sum(F_left)
        bright_right = np.sum(F_right)
        bright_diff = bright_left - bright_right
        Position = "center"
        if bright_diff < center_ratio - .05:
            Position = "right"
        if bright_diff > center_ratio + .05:
            Position = "left"
            
        
        
    # PRESENTITIONALS
    ## OpenCV presentitionals work different to PyGame. 
    ## In PG we are adding objects to a surface object. In OpenCV, the frame is augmented.
    
    if Detected:
        if Position == "center":
            font_col = (255,0,0)
        elif Position == "left":
            font_col = (0,0,255)
        elif Position == "right":
            font_col = (0,255,0)
        
        F_out = cv2.putText(F_eye, f"{bright_diff}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
        F_out = cv2.putText(F_out, f"{Position}", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2,
        font_col, thickness = 8)
    else:
        F_out = Frame
    
    cv2.imshow('YETI_2', F_out)
        
    
    # INTERACTIVE TRANSITIONALS
    ## Event handling is very limited in OpenCV
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# When everything is done, release the capture
YET.release()
cv2.destroyAllWindows()

