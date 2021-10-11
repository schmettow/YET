## YETI 3: Measuring the brightness distribution
## uses PyGame for interaction programming
## Output: Raw data brightness values per horizontal position

YETI = 3
YETI_NAME = "Yeti" + str(YETI)
TITLE = "Measuring the vertical brightness gradient"
AUTHOR = "M Schmettow"

import sys
import logging as log
import datetime as dt
from time import time
import random
# DS
import numpy as np
import csv
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
if YET.isOpened():
    width = int(YET.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(YET.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps =  YET.get(cv2.CAP_PROP_FPS)
    dim = (width, height)
else:
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
pg.display.set_caption("YETI_3: Measuring horizontal brightness distribution")
FONT = pg.font.Font('freesansbold.ttf',40)

SCREEN = pg.display.get_surface()
#SCREEN.fill(BACKGR_COL)

font = pg.font.Font(None, 60)


def main():
    
    ## Initial State
    STATE = "Prepare" # Stimulus, Measure
    DETECTED = False
    BACKGR_COL = col_black
    X_Stim = []
    Bright_T = []
    Bright_B = []
    Bright_diff = []
    Eyes = []

    ## FAST LOOP
    while True:
        pg.display.get_surface().fill(BACKGR_COL) 

        # General frame processing
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
            DETECTED = True
            (x, y, w, h) = Eyes[0]
            F_eye = F_gray[y:y+h,x:x+w]
        else: 
            DETECTED = False

        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Stimulus":
                if event.type == KEYDOWN and event.key == K_RETURN:
                    STATE = "Save"
                    print(STATE)
                elif event.type == KEYDOWN and event.key == K_SPACE:
                    if DETECTED:
                        STATE = "Measure"
                        print(STATE)
            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()
        

        # Automatic transitionals
        if STATE == "Prepare":
            Y = random.uniform(0,1) * SCREEN_SIZE[1]
            STATE = "Stimulus"
        elif STATE == "Measure":
            F_top =  F_eye[0:int(h/2), 0:w]
            F_bot = F_eye[int(h/2):h, 0:w]
            bright_top = np.mean(F_top)
            bright_bot = np.mean(F_bot)
            print(str(Y) + ": " + str(bright_top) + ")--(" + str(bright_bot))
            bright_diff = bright_top - bright_bot
            X_Stim.append(Y)
            Bright_T.append(bright_top)
            Bright_B.append(bright_bot)
            Bright_diff.append(bright_diff)
            STATE = "Prepare"
        elif STATE == "Save":
            myfile =  open("Brightness.csv", mode = "w")
            writer = csv.writer(myfile)
            for i in range(0, len(X_Stim)):
                thisrow = [X_Stim[i], Bright_T[i], Bright_B[i], Bright_diff[i]]
                writer.writerow(thisrow)
            myfile.close()


        # Presentitionals
        if STATE == "Stimulus":
            BACKGR_COL = col_black
            if DETECTED:
                Img = frame_to_surf(F_eye, (90, 70))
                draw_circ(SCREEN_SIZE[0]/2, Y, 20)
            else:
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

def draw_circ(x, y, radius, 
              color = (255,255,255), 
              stroke_size = 1):
    pg.draw.circle(SCREEN, color, 
                       (x,y), radius, stroke_size)

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
