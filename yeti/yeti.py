import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(2)
detected = False

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

    if not detected:
        eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200)
        )

        if len(eyes) > 0:
            eye = eyes[0]
            detected = True

    if detected:
        (x, y, w, h) = eye
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        eye_frame = gray[y:y+h,x:x+w]
        half_w = int(w/2)
        left_frame = eye_frame[y:y+h, x:x+half_w]
        cv2.putText(left_frame, #numpy array on which text is written
            "left", #text
            (50, 50), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            50, #font size
            (209, 80, 0, 255), #font color
            3) #font stroke
        right_frame = eye_frame[y:y+h, half_w:x]
        cv2.imshow('YETI', left_frame)
    else: 
        cv2.imshow('YETI', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
