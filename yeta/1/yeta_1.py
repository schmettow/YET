## YETA 1 Slideshow with YET
## Input: Images and PictureInfo.csv file
## Results = table with Part, Picture, time, eye tracking coordinates

import sys
import os
import time
import logging as log
import yeti14
from Stimulus import Stimulus, StimulusSet

"""GLOBAL VARIABLES"""

USB = 1
"""USB device for YET (typically 1, sometimes 0 or 2)"""

## Experiment
EXP_ID = "UV22"
"""Identifier for experiment"""
EXPERIMENTER = "MS"
"""Experimenter ID"""
SCREEN_W = 800
SCREEN_H = 800
"""Screen dimensions"""
SLIDE_TIME = 0.5
"""Presentation time per stimulus"""

## Paths and files
WD = os.path.dirname(sys.argv[0])
os.chdir(WD)
"""Working directory set to location of yeta_1.py"""

STIM_PATH = os.path.join(WD, "Stimuli")
"""Directory where stimuli reside"""
STIM_INFO = os.path.join(STIM_PATH, "Stimuli_short.csv")
"""CSV file describing stimuli"""
RESULT_DIR = "Data"
"""Directory where results are written"""
PART_ID = str(int(time.time()))
"""Unique participant identifier by using timestamps"""
RESULT_FILE = os.path.join(RESULT_DIR, "yeta1_" + EXP_ID + EXPERIMENTER + PART_ID + ".csv")
"""File name for data"""

## Meta data of Yeta
YETA = 1
YETA_NAME = "Yeta" + str(YETA)
TITLE = "Yeta 1: Slideshow"
AUTHOR = "M Schmettow, N Bierhuizen, GOF5(2021)"
EYECASC = "haarcascade_eye.xml"


import numpy as np
"""Numpy for data manipulation"""
from sklearn import linear_model as lm
"""Using linear models from Sklearn"""
import pandas as pd
"""Using Pandas data frames"""
import csv
"""Reading and writing CSV files"""


# CV
import cv2 as cv
"""OpenCV computer vision library"""

# PG
import pygame as pg
from pygame.locals import *
"""PyGame for the user interface"""

##### Preparations #####

# CV
log.basicConfig(filename='YET.log', level=log.INFO)


# Colour definitions
col_black = (0, 0, 0)
col_gray = (120, 120, 120)
col_red = (250, 0, 0)
col_green = (0, 255, 0)
col_yellow = (250, 250, 0)
col_white = (255, 255, 255)
col_red_dim = (120, 0, 0)
col_green_dim = (0, 60, 0)
col_yellow_dim = (120, 120, 0)

## width and height in pixel
SCREEN_SIZE = (SCREEN_W, SCREEN_H)

pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(YETA_NAME)
FONT = pg.font.Font('freesansbold.ttf', int(20 * min(SCREEN_SIZE) / 800))
Font = pg.font.Font('freesansbold.ttf', int(15 * min(SCREEN_SIZE) / 800))

SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 60)


def main():
    print("MAIN")
    YET = yeti14.YET(USB)
    print("FPS " + str(YET.fps))
    if(not YET.connected):
        print("Not connected")
        sys.exit()

    # Reading picture infos
    if os.path.isfile(STIM_INFO):
        STIMS = StimulusSet(STIM_INFO)
        print(str(STIMS.n()) + " stimuli loaded")
    else:
        print(STIM_INFO  + ' not found. CWD: ' + os.getcwd())
        sys.exit()

    ## Calibration screens
    CAL = yeti14.Calib(SCREEN)
    print("Calibration screen loaded with " + str(CAL.n()))
    QCAL = yeti14.Calib(SCREEN, rel_positions=[0.5])


    ## Initial State
    STATE = "Detect" 
    BACKGR_COL = col_white
    
    YET.init_eye_detection(EYECASC)
    print("YET connected: " + str(YET.connected))

    ## FAST LOOP
    while True:
        # FRAME PROCESSING
        YET.update_frame() ## a bit too general
        if STATE == "Detect":
            YET.detect_eye()
        
        if YET.eye_detected:
            YET.update_eye_frame()
        
        if STATE == "Validate" or STATE == "Stimulus" or STATE == "Quick":
            YET.update_quad_bright()
            YET.update_eye_pos()

        if STATE == "Stimulus":
            # YET.record(EXP_ID + EXPERIMENTER, PART_ID, this_image)
            YET.record(EXP_ID + EXPERIMENTER, PART_ID, this_stim.file)

        ## EVENT HANDLING
        for event in pg.event.get():
            key_forward = event.type == KEYDOWN and event.key == K_SPACE
            key_back = event.type == KEYDOWN and event.key == K_BACKSPACE
            key_return = event.type == KEYDOWN and event.key == K_RETURN
            
            # Interactive transition conditionals (ITC)
            if STATE == "Detect":
                if YET.eye_detected:
                    if key_forward:
                        CAL.reset()
                        STATE = "Target"
            elif STATE == "Target":
                if key_forward:
                    YET.update_eye_frame()
                    YET.update_quad_bright()
                    YET.record_calib_data(CAL.active_pos())
                    if CAL.remaining():
                        CAL.next()
                        STATE = "Target"
                    else:
                        YET.train()
                        STATE = "Validate"
                    print(STATE)
                elif key_back:
                    STATE = "Detect"
            elif STATE == "Validate":
                if key_forward:
                    STATE = "prepareStimulus"
                elif key_back:
                    CAL.reset()
                    YET.reset()
                    STATE = "Target"
            elif STATE == "Quick":
                if key_forward:
                    YET.update_offsets(QCAL.active_pos())
                    STATE = "prepareStimulus"
            elif STATE == "Thank You":
                if key_forward:
                    YET.release()
                    pg.quit()
                    sys.exit()
            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()



        # Automatic transitionals (ATC)
        if STATE == "prepareStimulus":
            ret, this_stim = STIMS.next()
            this_stim.load()
            if ret:
                STATE = "Stimulus"
                t_stim_started = time.time()
            
        if STATE == "Stimulus":
            elapsed_time = time.time() - t_stim_started # elapsed time since STATE == "Stimulus" started
            if elapsed_time > SLIDE_TIME: #presented longer then defined: trial is over
                YET.data.to_csv(RESULT_FILE, index = False) ## auto save results after every slide
                if STIMS.remaining() > 0:  # if images are left, got to quick cal
                    STATE = "Quick"
                    print(STATE)    
                else:
                    STATE = "Thank You"
                    print(STATE)
    

        # Presentitionals
        # over-paint previous with background xcolor
        pg.display.get_surface().fill(BACKGR_COL)

        if STATE == "Detect":
            if YET.eye_detected:
                Img = frame_to_surf(YET.eye_frame, (int(SCREEN_SIZE[0] * .5), int(SCREEN_SIZE[1] * .5)))
                msg = "Eye detected press Space to continue."
                draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .85), color=col_green)
            else:
                Img = frame_to_surf(YET.frame, (int(SCREEN_SIZE[0] * .5), int(SCREEN_SIZE[1] * .5)))
                msg = "Trying to detect an eye."
                draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .85), color=col_green)
            SCREEN.blit(Img, (int(SCREEN_SIZE[0] * .25), int(SCREEN_SIZE[1] * .25)))
        elif STATE == "Target":
            CAL.draw()
            # draw_target(SCREEN, targets, active_target)
            msg = "Follow the orange light and press Space."
            draw_text(msg, (SCREEN_SIZE[0]* .1, SCREEN_SIZE[1] * .75), color=col_green)
        elif STATE == "Validate":
            draw_validate(YET.eye_pos)
        elif STATE == "Stimulus":
            this_stim.show(SCREEN)
            # SCREEN.blit(IMG, (0, 0))
        elif STATE == "Quick":
            # draw_target(SCREEN, targets, active=4)
            QCAL.draw()
            msg = "Look at the orange circle and press Space."
            draw_text(msg, (SCREEN_SIZE[0] * .05, SCREEN_SIZE[1] * .75), color=col_green)
        elif STATE == "Thank You":
            draw_thank_you()
        # update the screen to display the changes you made
        pg.display.update()


    



## Converts a cv framebuffer into Pygame image (surface!)
def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img)  # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


def draw_text(text, dim,
              color=(255, 255, 255),
              center=False):
    x, y = dim
    rendered_text = FONT.render(text, True, color)
    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()
    # this sets the x and why coordinates
    if center:
        box.center = (x, y)
    else:
        box.topleft = (x, y)
    # This puts a pre-rendered object to the screen
    SCREEN.blit(rendered_text, box)


def draw_rect(x, y,
              width, height,
              color=(255, 255, 255),
              stroke_size=1):
    pg.draw.rect(SCREEN, color, (x, y, width, height), stroke_size)


def draw_thank_you():
    text_surface = FONT.render("Thank You", True, col_white, col_black)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] * 0.5, SCREEN_SIZE[1] * 0.3)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = FONT.render("For participation", True, col_white, col_black)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] * 0.5, SCREEN_SIZE[1] * 0.5)
    SCREEN.blit(text_surface, text_rectangle)


def draw_validate(eye_pos):
    eyex, eyey = eye_pos
    msg = "Press Space to continue the experiment."
    draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .75), color=col_green)
    msg = "Press Backspace to redo the calibration and reset the experiment."
    draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .8), color=col_green)
    draw_rect(eyex, 0, 2, SCREEN_H, stroke_size=1, color=col_green)
    draw_rect(0, eyey, SCREEN_W, 2, stroke_size=1, color=col_green)
    # diagnostics

    draw_text("HPOS: " + str(np.round(eyex)), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .15), color=col_green)
    draw_text("VPOS: " + str(np.round(eyey)), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .2), color=col_green)

main()
