## Yeti8: Two point calibration

# Experiments with Yeti2 showed, that teh split-frame brighness gradient
# is linearly related to horizontal eye ball position.
# This Yeti shows a quick calibration based on only two points

YETI = 8
YETI_NAME = "Yeti" + str(YETI)
TITLE = YETI_NAME + ": Quick calibration and follow"
AUTHOR = "M Schmettow"

DEBUG = False

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


##### DEFINITIONS ####

## width and height in pixel
SCREEN_SIZE = (2000, 500)
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]
HPOS  = (40, SCREEN_WIDTH - 40) ## x points for measuring

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
col_blue = (0, 0, 255)
col_yellow = (250, 250, 0)
col_white = (255, 255, 255)

col_red_dim = (120, 0, 0)
col_green_dim = (0, 60, 0)
col_yellow_dim = (120,120,0)


# initialize Pygame
pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(TITLE)

FONT = pg.font.Font('freesansbold.ttf',40)
SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 15)


def main():    
    ## Initial State
    STATE = "Detect" #  Measure_L, Measure_R, Follow, Save
    Detected = False
    Eyes = []
    SBD = 0
    H_offset = 0

    
    ## FAST LOOP
    while True:
        # General Frame Processing
        ret, Frame = YET.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # collecting a grayscale frame
        F_gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

        # Conditional Frame Processing
        if STATE == "Detect":
            # using the eye detection model on the frame
            Eyes = eyeCascade.detectMultiScale(
                F_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(200, 200))
            # when eye is detected, set the trigger, store the rectangle and get the subframe
            if len(Eyes) > 0:
                Detected = True
                (x, y ,w , h) = Eyes[0]
                F_eye = F_gray[y:y+h,x:x+w]
        # in all other states, we only extract teh eye frane and compute SBG_diff
        else:
            F_eye = F_gray[y:y+h,x:x+w]
            SBD = calc_SBD(F_eye)

        ## Event handling
        for event in pg.event.get():
            if event.type == KEYDOWN:
                # Interactive transition conditionals (ITC)
                if STATE == "Detect":
                    if event.key == K_SPACE:
                        if Detected:
                            STATE = "Measure_L"
                elif STATE == "Measure_L":
                    if event.key == K_SPACE:
                        SBD_L = SBD #collecting left SBG_diff
                        if DEBUG: print(SBD_L)
                        STATE = "Measure_R"
                    elif event.key == K_BACKSPACE:
                        STATE = "Detect"
                elif STATE == "Measure_R":
                    if event.key == K_SPACE:
                        SBD_R = SBD #collecting right SBG_diff
                        STATE = "Train" # Automkatic Transitional
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_L"
                elif STATE == "Validate":
                    if event.key == K_LEFT:
                        H_offset -= 5
                    if event.key == K_RIGHT:
                        H_offset += 5
                    if event.key == K_DOWN:
                        H_offset = 0
                    if event.key == K_SPACE:
                        write_csv(SBD_coef) # save coefficients 
                        STATE = "Save"
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_R"
                elif STATE == "Save":
                    if event.key == K_BACKSPACE:
                        STATE = "Validate"
                    elif event.key == K_SPACE:
                        STATE = "Detect"   ## ring closed
                print(STATE)

            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()
                


        # Automatic transitionals
        if STATE == "Train":
            SBD_coef = train_SBD((SBD_L, SBD_R), HPOS) # fitting the model
            print("Model:" + str(SBD_L) + str(SBD_R) + str(SBD_coef))
            STATE = "Validate"
            print

        # Presentitionals
        BACKGR_COL = col_white
        SCREEN.fill(BACKGR_COL)
        
        
        # YET stream
        if Detected:
            Img = frame_to_surf(F_eye, (400, 400))
        else:
            Img = frame_to_surf(F_gray, (400, 400))
        SCREEN.blit(Img,((SCREEN_WIDTH - 400)/2, (SCREEN_HEIGHT - 400)/2))
        
        if STATE == "Detect":
            BACKGR_COL = col_red_dim
            msg = "When eye is detected, press Space"

        if STATE == "Measure_L":
            msg = "Focus on the circle to your Left and press Space.  Backspace for back."
            draw_circ(HPOS[0], SCREEN_HEIGHT/2, 20, stroke_size=10, color = col_red)

        if STATE == "Measure_R":
            msg = "Focus on the circle to your Right and press Space. Backspace for back."
            draw_circ(HPOS[1], SCREEN_HEIGHT/2, 20, color = col_red, stroke_size=10)

        if STATE == "Validate":
            msg = "Press Space for saving.  Backspace for back."
            H_pos = predict_HPOS(SBD, SBD_coef)
            
            #draw_circ(H_pos + H_offset, SCREEN_SIZE[1]/2, 40 ,  stroke_size=10, color = col_blue)
            draw_rect(H_pos + H_offset - 2, 0, 4, SCREEN_HEIGHT, stroke_size=10, color = col_blue)
            draw_text("COEF: " + str(np.round(SBD_coef, 2)), (500, 50), color=col_black)
            draw_text("SBD : " + str(np.round(SBD)), (500, 150), color=col_black)
            draw_text("HPOS: " + str(np.round(H_pos)), (500, 250), color=col_black)
            draw_text("XOFF: " + str(H_offset), (500, 300), color=col_black)
        
        if STATE == "Saved":
            msg = "SBG.csv saved. Backspace for back.  Space for new cycle"
                            
        # Fixed UI elements
        draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .9), color=col_black)

        # update the screen to display the changes you made
        pg.display.update()


# splits a frame horizontally
def split_frame(Frame):
    height, width = Frame.shape
    F_left =  Frame[0:height, 0:int(width/2)]
    F_right = Frame[0:height, int(width/2):width]
    return F_left, F_right

# Computes the split-frame brightness diff
def calc_SBD(Frame):
    F_left, F_right = split_frame(Frame)
    bright_left = np.mean(F_left)
    bright_right = np.mean(F_right)
    sbd = bright_right - bright_left
    return sbd


# Estimates linear coefficients from two points and their brightness diff
def train_SBD(SBD, X):
    beta_1 = (X[1] - X[0])/(SBD[1] - SBD[0])
    beta_0 = X[0] - SBD[0] * beta_1 
    return (beta_0, beta_1)


# Predicts x position based on SBD and SBD coefficients
def predict_HPOS(sbd, coef):
    H_pos = coef[0] + sbd * coef[1]
    return H_pos


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

def draw_rect(x, y, 
              width, height, 
              color = (255,255,255), 
              stroke_size = 1):
    pg.draw.rect(SCREEN, color, (x, y, width, height), stroke_size)

def write_csv(coef):
    pass

main()
