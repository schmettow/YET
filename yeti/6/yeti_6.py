## YETI_6: Blink detection and duration measures

YETI = 6
YETI_NAME = "Yeti" + str(YETI)
TITLE = "Blink detection"
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

##### Preparations #####

# CV

log.basicConfig(filename='YET.log',level=log.INFO)
YET = cv2.VideoCapture(1)
if not YET.isOpened():
        print('Unable to load camera.')
        exit()
eyeCascade = cv2.CascadeClassifier("./trained_models/haarcascade_eye.xml")

# PG

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

pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption("YETI_6: Simple blink detection")
FONT = pg.font.Font('freesansbold.ttf',40)

SCREEN = pg.display.get_surface()
#SCREEN.fill(BACKGR_COL)

font = pg.font.Font(None, 60)


def main():
    
    ## Initial State
    STATE = "Start"
    last_Detected = False
    Detected = False
    T_Open = 0
    T_Blink = 0
    N_Open = 0
    N_Blink = 0
    Colors = (col_red, col_yellow, col_green)
    
    ## FAST LOOP
    while True:
        # Frame processing
        ret, Frame = YET.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        F_gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        Eyes = eyeCascade.detectMultiScale(
                F_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(200, 200))
        
        # Conditional frame processing
        if len(Eyes) > 0:
            (x, y, w, h) = Eyes[0]
            last_Detected = Detected
            Detected = True
            if STATE == "Measure":
                F_out = Frame[y:y+h,x:x+w]
            elif STATE == "Start":
                F_out = cv2.rectangle(Frame, (x, y), (x + w, y + h) , (0, 255, 0), 2)

        else:
            F_out = Frame
            last_Detected = Detected
            Detected = False
        
        ## Detecting a blink start and end
        if STATE == "Measure" and not Detected == last_Detected:
            if not Detected: # blink start
                T_Closed = time()
                T_Open = T_Opened - T_Closed
                N_Blink = N_Blink + 1
            if Detected:     # blink end
                T_Opened = time()
                T_Blink = T_Closed - T_Opened
                N_Open += 1

        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if event.type == KEYDOWN and event.key == K_SPACE and Detected:
                if STATE == "Start":
                    STATE = "Measure"
                    T_Opened = time()
                    print(STATE)
                elif STATE == "Measure":
                    STATE = "Start"
                    print(STATE)
            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()
            

        # Presentitionals
        if STATE == "Start":
            BACKGR_COL = col_white
        if STATE == "Measure":
            BACKGR_COL = Colors[N_Blink % len(Colors)] ## modulo to cycle through colors
        SCREEN.fill(BACKGR_COL)
        draw_text("Last blink (s): " + str(round(T_Blink, 2)), (10, 10))
        Img = frame_to_surf(F_out, (900, 700))
        SCREEN.blit(Img,(50,50))
        
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
