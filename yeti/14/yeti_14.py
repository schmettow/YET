## YETI 14: Collecting 2-dim calibration data for quadrant SBG
## input = calibration point table
## Results = table with target coordinates and quadrant brightness

from time import time

YETI = 14
YETI_NAME = "Yeti" + str(YETI)
TITLE = "Multi-point calibration measures with quadrant brightness"
AUTHOR = "M SCHMETTOW"
CONFIG = "14/yeti_14.xlsx"
RESULTS = "14/yeti_14_" + str(time()) + ".xlsx"

import sys
import os
import logging as log

# DS
import datetime as dt
import random
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
import csv
# CV
import cv2 as cv
# PG
import pygame as pg
from pygame.locals import *
from pygame.compat import unichr_, unicode_

##### Preparations #####

# CV
log.basicConfig(filename='YET.log',level=log.INFO)
YET = cv.VideoCapture(1)
if YET.isOpened():
    width = int(YET.get(cv.CAP_PROP_FRAME_WIDTH ))
    height = int(YET.get(cv.CAP_PROP_FRAME_HEIGHT ))
    fps =  YET.get(cv.CAP_PROP_FPS)
    dim = (width, height)
else:
    print('Unable to load camera.')
    exit()

# Reading the CV model for eye detection
eyeCascadeFile = "../trained_models/haarcascade_eye.xml"
if os.path.isfile(eyeCascadeFile):
    eyeCascade = cv.CascadeClassifier(eyeCascadeFile)
else:
    sys.exit(eyeCascadeFile + ' not found. CWD: ' + os.getcwd())

# Reading calibration point matrix from Excel table
Targets = pd.read_excel(CONFIG)

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
SCREEN_W = 1000
SCREEN_H = 1000
SCREEN_SIZE = (SCREEN_W, SCREEN_H)


pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(YETI_NAME)
FONT = pg.font.Font('freesansbold.ttf',40)

SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 60)


def main():

    ## Initial State
    STATE = "Detect" # Measure, Target
    DETECTED = False
    BACKGR_COL = col_black
    
    n_targets = len(Targets)
    active_target = 0
    run = 0
    H_offset, V_offset = (0,0)
    this_pos = (0,0)

    Eyes = []
    OBS_cols = ("Obs", "run", "NW", "NE", "SW", "SE", "x", "y")
    OBS = np.empty(shape = (0, len(OBS_cols)))
    obs = 0
    

    ## FAST LOOP
    while True:
        # General frame processing
        ret, Frame = YET.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        F_gray = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)

        # Eye detection. Eye frame is being locked.

        if STATE == "Detect":
            Eyes = eyeCascade.detectMultiScale(
                    F_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100))
            if len(Eyes) > 0:
                DETECTED = True
                x_eye, y_eye, w_eye, h_eye = Eyes[0]
                F_eye = F_gray[y_eye:y_eye+h_eye,x_eye:x_eye+w_eye]
            else: 
                DETECTED = False
                # F_eye = F_gray
                # w_eye, h_eye = (width, height)
        else:
            F_eye = F_gray[y_eye:y_eye+h_eye,x_eye:x_eye+w_eye]
            
        if STATE == "Validate":
            this_quad = np.array(quad_bright(F_eye))
            this_quad.shape = (1,4)
            this_pos = M_0.predict(this_quad)[0,:]
            print(this_pos)
            

        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Detect":
                if DETECTED:
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        this_Eye = Eyes[0]
                        x_eye, y_eye, w_eye, h_eye = this_Eye
                        STATE = "Target"
                        print(STATE  + str(active_target))
                        
            elif STATE == "Target":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Measure"
                    print(STATE  + str(active_target))
            elif STATE == "Save":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Train" 
                if event.type == KEYDOWN and event.key == K_RETURN:
                    STATE = "Save" 
                    OBS = pd.DataFrame(OBS, columns = OBS_cols)
                    with pd.ExcelWriter(RESULTS) as writer:
                        print(OBS)
                        OBS.to_excel(writer, sheet_name="Obs_" + str(time()), index = False)
                        print(RESULTS)
                if event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target" 
                    active_target = 0 # reset
                    run = run + 1
        if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()
        

        # Automatic transitionals
        if STATE == "Measure":
            obs = obs + 1
            this_id = (obs, run)
            this_targ = tuple(Targets.to_numpy()[active_target][0:2])
            this_bright = quad_bright(F_eye)
            #print(this_targ.shape, this_id.shape, this_bright.shape)
            this_obs = this_id + this_bright + this_targ
            print(this_obs)
            OBS = np.vstack((OBS, this_obs))
            
            if (active_target + 1) < n_targets:
                active_target = active_target + 1
                STATE = "Target"
                print(STATE  + str(active_target))
            else:
                print(OBS)
                STATE = "Save"
        
        if STATE == "Train":
            M_0 = train_QBG(OBS)
            STATE = "Validate"

        # Presentitionals
        # over-paint previous with background xcolor
        pg.display.get_surface().fill(BACKGR_COL)

        if STATE == "Detect":
            if DETECTED:
                Img = frame_to_surf(F_eye, (200, 200))
            else:
                Img = frame_to_surf(Frame, (200, 200))
            SCREEN.blit(Img,(400,400))
        elif STATE == "Target":
            if DETECTED:
                drawTargets(SCREEN, Targets, active_target)
        elif STATE == "Validate":
            msg = "Press Space one time to stop validating, two times for saving. Backspace for back."
            draw_rect(this_pos[0] + H_offset - 1, 0, 2, SCREEN_H, stroke_size=1, color = col_green)
            draw_rect(0, this_pos[1] + V_offset - 1, SCREEN_W, 2, stroke_size=1, color = col_green)
            # diagnostics
            draw_text("HPOS: " + str(np.round(this_pos[0])), (510, 250), color=col_green)
            draw_text("VPOS: " + str(np.round(this_pos[1])), (510, 300), color=col_green)
            draw_text("XOFF: " + str(H_offset), (510, 350), color=col_green)
            draw_text("YOFF: " + str(V_offset), (510, 400), color=col_green)
        
        
        # update the screen to display the changes you made
        pg.display.update()


def drawTargets(screen, Targets, active = 0):
    for index, target in Targets.iterrows():
        color = (160,160,160)
        radius = 20
        stroke = 10
        if index == active: color = (255,120,0)
        pos = (target['x'], target['y'])
        pg.draw.circle(screen, color, pos, radius, stroke)

## Converts a cv framebuffer into Pygame image (surface!)
def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB) # convert BGR (cv) to RGB (Pygame)
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

def draw_rect(x, y, 
              width, height, 
              color = (255,255,255), 
              stroke_size = 1):
    pg.draw.rect(SCREEN, color, (x, y, width, height), stroke_size)


def quad_bright(frame):
    w, h = np.shape(frame)
    b_NW =  np.mean(frame[0:int(h/2), 0:int(w/2)])
    b_NE =  np.mean(frame[int(h/2):h, 0:int(w/2)])
    b_SW =  np.mean(frame[0:int(h/2), int(w/2):w])
    b_SE =  np.mean(frame[int(h/2):h, int(w/2):w])
    out = (b_NW, b_NE, b_SW, b_SE)
    # out = np.array((b_NW, b_NE, b_SW, b_SE))
    # out.shape = (1,4)
    return(out)

def train_QBG(Obs):
    Quad = Obs[:,2:6]
    Pos = Obs[:,6:8]
    model = lm.LinearRegression()
    model.fit(Quad, Pos)
    return model


# Predicts position based on quad-split
def predict_pos(data, model):
    predictions = model.predict(data)
    return predictions

main()
