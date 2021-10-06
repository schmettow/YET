## Yeti8: Two point calibration

# Experiments with Yeti2 showed, that teh split-frame brighness gradient
# is linearly related to horizontal eye ball position.
# This Yeti shows a quick calibration based on only two points


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

log.basicConfig(filename='YET.log',level=log.INFO)
YET = cv2.VideoCapture(1)
if not YET.isOpened():
        print('Unable to load camera.')
        exit()

# Reading the CV model for eye detection
eyeCascade = cv2.CascadeClassifier("./trained_models/haarcascade_eye.xml")

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
pg.display.set_caption("Yeti8: quick calibration and follow")

FONT = pg.font.Font('freesansbold.ttf',20)
SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 15)


def main():    
    ## Initial State
    STATE = "Detect" #  Measure_L, Measure_R, Follow, Save
    Detected = False
    Eyes = []

    
    ## FAST LOOP
    while True:
        # General Frame Processing
        ret, Frame = YET.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # cvollecting a grayscale frame
        F_gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

        # Conditional Frame Processing
        if STATE == "Detect":
            # using the eye detection model
            Eyes = eyeCascade.detectMultiScale(
                F_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(200, 200))
            # when eye is detected, change state and store coordinates
            if len(Eyes) > 0:
                Detected = True
                (x, y ,w , h) = Eyes[0]
                F_eye = F_gray[y:y+h,x:x+w]
        # in all other states, we only extract teh eye frane and compute SBG_diff
        else:
            F_eye = F_gray[y:y+h,x:x+w]
            sbg_diff = SBG_diff(F_eye)

        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Detect":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    if Detected:
                        STATE = "Measure_L"
                        print(STATE)
            elif STATE == "Measure_L":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Measure_R"
                    sbg_L = sbg_diff #collecting left SBG_diff
                    print(STATE)
            elif STATE == "Measure_R":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Follow"
                    sbg_R = sbg_diff #collecting left SBG_diff
                    sbg_coef = SBG_fit(sbg_L, sbg_R, 10, 990) # fitting the model ;)
                    print(STATE)
            elif STATE == "Follow":
                if event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        write_csv(sbg_coef) # save coefficients 
                        STATE = "Saved"
                    elif event.key == K_RETURN:
                        STATE = "Measure_L"  
                    print(STATE)
            elif STATE == "Saved":
                if event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        STATE = "Follow"
                    elif event.key == K_RETURN:
                        STATE = "Measure_L"

            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()


        # Automatic transitionals
        pass
                

        # Presentitionals
        BACKGR_COL = col_black
        SCREEN.fill(BACKGR_COL)
        
        if Detected:
            Img = frame_to_surf(F_eye, (200, 200))
        else:
            Img = frame_to_surf(F_gray, (200, 200))

        if STATE == "Detect":
            BACKGR_COL = col_red_dim
            msg = "When eye is detected, press Space"

        if STATE == "Measure_L":
            msg = "Focus on the circle to your Left and press Space"
            draw_circ(10, SCREEN_SIZE[1]/2, 20, stroke_size=5)

        if STATE == "Measure_R":
            msg = "Focus on the circle to your Right and press Space"
            draw_circ(SCREEN_SIZE[0]-10, SCREEN_SIZE[1]/2, 20, stroke_size=5)

        if STATE == "Follow":
            msg = "Space for saving, Return for re-calibration."
            x_pos = SBG_predict(sbg_diff, sbg_coef)
            draw_circ(x_pos, SCREEN_SIZE[1]/2, 40 ,  stroke_size=10, color=(0, 255, 255))
        
        if STATE == "Saved":
            msg = "SBG.csv saved. Space: follow, Return: re-calibration."
        
        # Fixed UI elements
        draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .9), color=col_gray)
        SCREEN.blit(Img,(50,50))
                    
        # update the screen to display the changes you made
        pg.display.update()


# Estimates the split-frame brightness diff
def SBG_diff(frame):
    height, width = frame.shape
    F_left =  frame[0:height, 0:int(width/2)]
    F_right = frame[0:height, int(width/2):width]
    bright_left = np.mean(F_left)
    bright_right = np.mean(F_right)
    bright_diff = bright_left - bright_right
    return bright_diff


# Estimates linear coefficients from two points and their brightness diff
def SBG_fit(diff_L, diff_R, margin_L, margin_R):
    beta_0 = diff_L
    beta_1 = (margin_R - margin_L)/(diff_R - diff_L)
    return (beta_0, beta_1)


# Predicts x position based on BGS diff and BGS coefficients
def SBG_predict(bright_diff, beta):
    return beta[0] + bright_diff * beta[1]


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

def draw_circ(x, y, radius, 
              color = (255,255,255), 
              stroke_size = 1):
    pg.draw.circle(SCREEN, color, (x,y), radius, stroke_size)


def write_csv(coef):
    pass

main()
