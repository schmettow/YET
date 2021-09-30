import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

log.basicConfig(filename='webcam.log',level=log.INFO)

YET = cv2.VideoCapture(1)

if not YET.isOpened():
        print('Unable to load camera.')
        exit()

while True:

    # Capture frame-by-frame
    ret, frame = YET.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('YETI', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# When everything is done, release the capture
YET.release()
cv2.destroyAllWindows()
