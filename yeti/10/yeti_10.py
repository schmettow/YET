## Yeti10: face and emotion

# Captures the cam stream 
# detects faces and emotions

YETI = 10
YETI_NAME = "Yeti" + str(YETI)
TITLE = "Face and emotion"
AUTHOR = "M Schmettow"

import sys
import logging as log
import datetime as dt
from time import time
import random
# DS
import numpy as np
# CV
import cv2
# PG
import pygame as pg
from pygame.locals import *
from pygame.compat import unichr_, unicode_

## Emotion detection
from fer import FER


##### Preparations #####

# PG
# color definitions
col_black = (0, 0, 0)
col_gray = (120, 120, 120)
col_red = (250, 0, 0)
col_green = (0, 255, 0)
col_yellow = (250, 250, 0)
col_white = (255, 255, 255)

col_red_dim = (120, 0, 0)
col_green_dim = (0, 60, 0)
col_yellow_dim = (120,120,0)

## width and height in pixel
SCREEN_SIZE = (1000, 800)

# initialize PG window
pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(YETI_NAME + ": " + TITLE)

FONT = pg.font.Font('freesansbold.ttf',20)
SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 15)

# Connecting to CAM stream

log.basicConfig(filename=YETI_NAME + '.log', level=log.INFO)

CAM = cv2.VideoCapture(0)
if not CAM.isOpened():
    print('Unable to load Webcam.')
    exit()

# CV
faceCascade = cv2.CascadeClassifier("./trained_models/haarcascade_frontalface_default.xml")
Emo_detect = FER(mtcnn=True)


def main():    
    ## Initial State
    STATE = "Stream" # Analyze
    FACE = False
    EMO = None

    
    ## FAST LOOP
    while True:
        # General Frame Processing
        ret, F_Cam = CAM.read(1)
        if ret:
            F_Cam_gray = cv2.cvtColor(F_Cam, cv2.COLOR_BGR2GRAY)
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
        
        # Conditional Frame Processing
        if STATE == "Stream":
            F_out = F_Cam
            FACE = False
            Faces_Cam = faceCascade.detectMultiScale(F_Cam_gray, scaleFactor=1.1,
                minNeighbors=5, minSize=(100, 100))
                # loop over the face bounding boxes
            for (fX, fY, fW, fH) in Faces_Cam:
                FACE = True
                # extract the face ROI
                F_Face = F_Cam[fY:fY+ fH, fX:fX + fW]
                # painting the face frame
                F_out = cv2.rectangle(F_out, (fX, fY), (fX + fW, fY + fH), (0, 255, 255), 2)
        ## No elif? Face detection is continuously. Emotion analysis is so slow,
        # it cannot be continuous. Emotional analysis is done only on 
        # a single snapshot. That's why it must be in the Transitionals, not here.

        ## Event handling
        for event in pg.event.get():
            if event.type == KEYDOWN and event.key == K_SPACE:
                if STATE == "Snapshot":
                    STATE = "Stream"
                elif STATE == "Stream":
                    if FACE:
                        # Creating the output frame for the snap shot
                        # emotion analysis on detected faces (frame frame processing)
                        for (fX, fY, fW, fH) in Faces_Cam:
                            F_Face = F_Cam[fY:fY+ fH, fX:fX + fW]
                            EMO, score = Emo_detect.top_emotion(frame_to_img(F_Face))
                            cv2.putText(F_Face, EMO, (fX,fY), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255))
                            print(EMO)
                    STATE = "Snapshot"
                print(STATE)
                

            
            if event.type == QUIT:
                CAM.release()
                pg.quit()
                sys.exit()
                


        # Automatic transitionals
        pass        

        # Presentitionals
        BACKGR_COL = col_white
        SCREEN.fill(BACKGR_COL)
        Img = frame_to_surf(F_out, (960, 720))
        SCREEN.blit(Img,(0,0))
        draw_text(EMO, (500, 100), center = True)
            
        # update the screen to display the changes you made
        pg.display.update()



## Converts a CV2 framebuffer into Pygame image (surface!)
def frame_to_img(frame):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # convert BGR (CV2) to RGB (Pygame)
    img = np.rot90(img) # rotate coordinate system
    return img

def frame_to_surf(frame, dim):
    img = frame_to_img(frame)
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf



def draw_text(text, dim,
              color = (255, 255, 255),
              center = False):
    x, y = dim
    rendered_text = FONT.render(text, True, color)
    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()
    # this sets the x and why coordinates
    if center:
        box.center = (x,y)
    else:
        box.topleft = (x,y)
    # This puts a pre-rendered object to the screen
    SCREEN.blit(rendered_text, box)

main()
