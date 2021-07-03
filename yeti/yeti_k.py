import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascadeFile = "haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascadeFile)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(2)

if not video_capture.isOpened():
        print('Unable to load camera.')
        exit()

threshold = 117

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200)
    )

    if len(eyes) > 0:
        (x, y, w, h) = eyes[0]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0))

        ret, thresh_gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        frame = cv2.drawContours(frame, contours, -1, (255,255,0))

        frame = cv2.putText(frame, f"contours: {len(contours)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,0) )
        frame = cv2.putText(frame, f"threshold: {threshold}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,0) )

        cv2.imshow('YETI', frame)
        cv2.imshow("grey", gray)
        cv2.imshow("thresholds", thresh_gray)
    else: 
        cv2.imshow('YETI', frame)
    
    key = cv2.waitKey(1)
    if key >= 0:
        if key & 0xFF == ord('q'):
            break
        if key == ord('+'):
            threshold = min( threshold + 10, 255 )
        if key == ord('-'):
            threshold = max( threshold - 10, 0 )

    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
