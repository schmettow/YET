## YETI_6: Simple blink detection and duration measures

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
eyeCascade = cv2.CascadeClassifier("./models/haarcascade_eye.xml")

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
    STATE = "Stream" # Detected, Closed
    BACKGR_COL = col_black
    T_Blink = 999.99
    
    ## FAST LOOP
    while True:
        pg.display.get_surface().fill(BACKGR_COL) 
        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Stream":
                pass
            elif STATE == "Detected":
                pass
            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()
            
        # Frame capturing and unconditional processing
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
        if len(Eyes) > 0:
            (x, y, w, h) = Eyes[0]
            F_eye = F_gray[y:y+h,x:x+w]

        # Automatic transitionals
        if STATE == "Stream":
            if len(Eyes) > 0:
                print("Eye detected")
                STATE = "Detected"
        elif STATE == "Detected":
            if len(Eyes) == 0:
                print("Closed")
                STATE = "Closed"
                T_Closed_on = time()
        elif STATE == "Closed":
            if len(Eyes) > 0:
                STATE = "Detected"
                T_Blink = time() - T_Closed_on
                print(round(T_Blink, 2))
        
        # Presentitionals
        if STATE == "Stream":
            BACKGR_COL = col_white
            Img = frame_to_surf(Frame, (900, 700))
            SCREEN.blit(Img,(50,50))

        if STATE == "Detected":
            BACKGR_COL = col_black
            Img = frame_to_surf(F_eye, (900, 700))
            SCREEN.blit(Img,(50,50))
            draw_text("Last blink (s): " + str(round(T_Blink, 2)), (10, 10))

        if STATE == "Closed":
            BACKGR_COL = col_red_dim
            Img = frame_to_surf(F_gray, (900, 700))
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
