## Yeti9: Dual cam viewer

# Captures two streams and displays them
# Space toggles eye/face detection mode
# Return toggles layout


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

##### Preparations #####

# Connecting to YET

log.basicConfig(filename='Yeti_9.log', level=log.INFO)

CAM = cv2.VideoCapture(0)
if not CAM.isOpened():
    print('Unable to load Webcam.')
    exit()

YET = cv2.VideoCapture(1)
if not YET.isOpened():
    print('Unable to load Yet.')
    exit()

# CV
eyeCascade = cv2.CascadeClassifier("./trained_models/haarcascade_eye_tree_eyeglasses.xml")
faceCascade = cv2.CascadeClassifier("./trained_models/haarcascade_frontalface_default.xml")

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
pg.display.set_caption("Yeti9: Dual tracking")

FONT = pg.font.Font('freesansbold.ttf',20)
SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 15)


def main():    
    ## Initial State
    STATE = "View" #  Detect
    Eyes = []
    LAYOUT = "Pair" # PIP

    
    ## FAST LOOP
    while True:
        # General Frame Processing
        ret, F_Yet = YET.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        ret, F_Cam = CAM.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Conditional Frame Processing
        if STATE == "Detect":
            # collecting a grayscale frame
            F_Yet_gray = cv2.cvtColor(F_Yet, cv2.COLOR_BGR2GRAY)
            F_Cam_gray = cv2.cvtColor(F_Cam, cv2.COLOR_BGR2GRAY)
            # using the eye detection model on Yet stream
            Eyes_Yet = eyeCascade.detectMultiScale(F_Yet_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(200, 200))
            # loop over the eyes bounding boxes
            for (eX, eY, eW, eH) in Eyes_Yet:
                # loop over the eye bounding boxes
                # draw the eye bounding box
                ptA = (eX, eY)
                ptB = (eX + eW, eY + eH)
                F_Yet = cv2.rectangle(F_Yet, ptA, ptB, (0, 0, 255), 2)
            # creating a hierarchical face-then-eyes detection
            Faces_Cam = faceCascade.detectMultiScale(F_Cam_gray, scaleFactor=1.1,
                minNeighbors=5, minSize=(100, 100))
            # loop over the face bounding boxes
            for (fX, fY, fW, fH) in Faces_Cam:
                # painting the face frame
                F_Cam = cv2.rectangle(F_Cam, (fX, fY), (fX + fW, fY + fH), (0, 255, 255), 2)
                # extract the face ROI
                F_Face = F_Cam_gray[fY:fY+ fH, fX:fX + fW]
                
                # apply eyes detection to the face ROI
                Eyes_Face =  eyeCascade.detectMultiScale(F_Face, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
                for (eX, eY, eW, eH) in Eyes_Face:
                    # looping over eyes in one face, hopefully two.
                    ptA = (fX + eX, fY + eY)
                    ptB = (fX + eX + eW, fY + eY + eH)
                    # painting the eye frame
                    F_Cam = cv2.rectangle(F_Cam, ptA, ptB, (0, 0, 255), 2)
            
            


        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            # Written in reverse! These are basically two buttons
            # This is why it is much shorter to go from event conditional
            # to state conditional, rather than the other way round, as  we used
            # to.
            if event.type == KEYDOWN and event.key == K_SPACE:    
                if STATE == "View":
                    STATE = "Detect"
                elif STATE == "Detect":
                    STATE = "View"
                    print(STATE)
            if event.type == KEYDOWN and event.key == K_RETURN:    
                if LAYOUT == "PIP":
                    LAYOUT = "Pair"
                elif LAYOUT == "Pair":
                    LAYOUT = "PIP"
                    print(LAYOUT)    
            if event.type == QUIT:
                CAM.release()
                YET.release()
                pg.quit()
                sys.exit()
                


        # Automatic transitionals
        pass        

        # Presentitionals
        BACKGR_COL = col_white
        SCREEN.fill(BACKGR_COL)

        # Conditional UI elements
        if STATE == "View":
            msg_1 = "View. Press Space."
        if STATE == "Detect":
            msg_1 = "Detect. Press Space."

        if LAYOUT == "Pair":
            msg_2 = "Pair. Press Return."
            Img_1 = frame_to_surf(F_Cam, (480, 320))
            Img_2 = frame_to_surf(F_Yet, (480, 320))
            SCREEN.blit(Img_1,(0,0))
            SCREEN.blit(Img_2,(500,0))
        if LAYOUT == "PIP":
            msg_2 = "Picture-in-picture. Press Return."
            Img_1 = frame_to_surf(F_Cam, (960, 720))
            Img_2 = frame_to_surf(F_Yet, (240, 180))
            SCREEN.blit(Img_1,(0,0))
            SCREEN.blit(Img_2,(720,540))
        
        # Static UI elements
        draw_text(msg_1, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .8), color=col_black)
        draw_text(msg_2, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .9), color=col_black)
            
        # update the screen to display the changes you made
        pg.display.update()



## Converts a CV2 framebuffer into Pygame image (surface!)
def frame_to_surf(frame, dim):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # convert BGR (CV2) to RGB (Pygame)
    img = np.rot90(img) # rotate coordinate system
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
