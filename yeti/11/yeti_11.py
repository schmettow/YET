## Yeti8: Two point calibration

# Experiments with Yeti2 showed, that teh split-frame brighness gradient
# is linearly related to horizontal eye ball position.
# This Yeti shows a quick calibration based on only two points

YETI = 11
YETI_NAME = "Yeti" + str(YETI)
TITLE = YETI_NAME + ": Quick calibration and follow"
AUTHOR = "M Schmettow"

DEBUG = False

import sys
import os.path
import logging as log
from datetime import datetime
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
SCREEN_SIZE = (1600, 1000)
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]
HPOS = (40, SCREEN_WIDTH - 40) ## x points for measuring
VPOS = (40, SCREEN_HEIGHT - 40)
xy_coords = []
timestamps = []

##### Preparations #####

# Reading the CV model for eye detection
eyeCascadeFile = "../trained_models/haarcascade_eye.xml"
if os.path.isfile(eyeCascadeFile):
    eyeCascade = cv2.CascadeClassifier(eyeCascadeFile)
else:
    sys.exit(eyeCascadeFile + ' not found. CWD: ' + os.getcwd())

# PG
# color definitions
col_black = (0, 0, 0)
col_gray = (120, 120, 120)
col_darkgray = (50, 50, 50)
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

Font = pg.font.Font('freesansbold.ttf',30)
Font_small = pg.font.Font('freesansbold.ttf',15)
SCREEN = pg.display.get_surface()

def main():    
    Yetstream = False
    ## Initial interaction State
    STATE = "welcome" #show welcome screen
    Detected = False # eye detection
    Eyes = [] # results of eye detection
    SBD = 0 # split frame difference
    H_offset = 0 # manual horizontal offset
    V_offset = 0 # manual vertical offset 
    y = 0
    x = 0
    w = 0
    h = 0

        # Connecting to YET
    YET = cv2.VideoCapture(1)
    if YET.isOpened():
        Yetstream = True
        width = int(YET.get(cv2.CAP_PROP_FRAME_WIDTH ))
        height = int(YET.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        fps =  YET.get(cv2.CAP_PROP_FPS)
        dim = (width, height)
        print('YET stream' + str(dim) + "@" + str(fps))
    else:
        print('Unable to load YET.')
        exit()



    
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
            else:
                Detected = False
        # in all other states, we only extract teh eye frane and compute SBG_diff
        else:
            F_eye = F_gray[y:y+h,x:x+w]
            F_left, F_right = split_frame(F_eye)
            SBD = calc_SBD(F_left, F_right)

        ## Event handling
        for event in pg.event.get():
            if event.type == KEYDOWN:
                # Interactive transition conditionals (ITC)
                ## Five consecutive steps with back-and-forth navigation
                if STATE == "welcome":
                    if event.key == K_SPACE:
                            STATE = "Detect"  
                elif STATE == "Detect":
                    if event.key == K_SPACE:
                        if Detected:
                            STATE = "Measure_L"
                    elif event.key == K_BACKSPACE:
                        STATE = "welcome"
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
                        STATE = "Measure_T" # Automkatic Transitional
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_L"
                elif STATE == "Measure_T":
                    if event.key == K_SPACE:
                        SBD_T = SBD #collecting top SBG_diff
                        if DEBUG: print(SBD_B)
                        STATE = "Measure_B"
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_R"
                elif STATE == "Measure_B":
                    if event.key == K_SPACE:
                        SBD_B = SBD #collecting bottom SBG_diff
                        STATE = "Train" # Automatic Transitional
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_T"
                elif STATE == "Validate":
                    print(xy_coords)
                    if event.key == K_LEFT:
                        H_offset -= 5
                    if event.key == K_RIGHT:
                        H_offset += 5
                    if event.key == K_DOWN:
                        V_offset -= 5
                    if event.key == K_UP:
                        V_offset += 5
                    if event.key == K_SPACE:
                        write_csv(SBD_model_h) # save coefficients
                        write_csv(SBD_model_v) # save coefficients 
                        STATE = "Save"
                    elif event.key == K_BACKSPACE:
                        STATE = "Detect"
                elif STATE == "Save":
                    now = datetime.now()
                    #dd/mm/YY H:M:S
                    date = now.strftime("%d_%m_%Y-%H_%M")
                    list_of_coordinates = np.array(xy_coords)
                    list_of_timestamps = np.array(timestamps)
                    list_of_all_data = [list_of_coordinates, list_of_timestamps]
                    print(list_of_coordinates)
                    np.savetxt(f"eyeresults_{date}.csv", list_of_all_data, fmt='%s')
    
                    if event.key == K_BACKSPACE:
                        STATE = "Validate"
                    elif event.key == K_RETURN:
                        STATE = "Detect"   ## ring closed
                print(STATE)

            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()
                


        # Automatic transitionals
        if STATE == "Train":
            SBD_model_h = train_SBD((SBD_L, SBD_R), HPOS) # fitting the model
            SBD_model_v = train_SBD((SBD_T, SBD_B), VPOS) # fitting the model
            print("SBD: " + str((SBD_L, SBD_R)) + "Model: " + str(SBD_model_h))
            print("SBD: " + str((SBD_T, SBD_B)) + "Model: " + str(SBD_model_v))
            STATE = "Validate"
            print(STATE)

        # Presentitionals
        BACKGR_COL = col_darkgray
        SCREEN.fill(BACKGR_COL)
        
        
        # YET stream
        
        Img = frame_to_surf(F_gray, (0, 0))
        
        if STATE != "welcome":
            
            Img = frame_to_surf(F_gray, (400, 400))
        SCREEN.blit(Img,((SCREEN_WIDTH - 400)/2, (SCREEN_HEIGHT - 400)/2))
        
        if STATE == "welcome":
            msg = "Welcome to this eyetracking experiment. Press space to continue"
            
        if STATE == "Detect":
            if Detected:
                msg = "Press Space. Backspace for back."
            else:
                msg = "Eye not detected. Backspace for back."

        if STATE == "Measure_L":
            msg = "Focus on the circle to your Left and press Space.  Backspace for back."
            draw_circ(HPOS[0], SCREEN_HEIGHT/2, 20, stroke_size=10, color = col_red)

        if STATE == "Measure_R":
            msg = "Focus on the circle to your Right and press Space. Backspace for back."
            draw_circ(HPOS[1], SCREEN_HEIGHT/2, 20, color = col_red, stroke_size=10)
            
        if STATE == "Measure_T":
            msg = "Focus on the circle at the Top and press Space.  Backspace for back."
            draw_circ(SCREEN_WIDTH / 2, VPOS[0], 20, stroke_size=10, color = col_red)

        if STATE == "Measure_B":
            msg = "Focus on the circle at the Bottom and press Space. Backspace for back."
            draw_circ(SCREEN_WIDTH / 2, VPOS[1], 20, color = col_red, stroke_size=10)

        if STATE == "Validate":
            msg = "Press Space one time to stop validating, two times for saving. Backspace for back."
            H_pos = predict_HPOS(SBD, SBD_model_h)
            V_pos = predict_VPOS(SBD, SBD_model_v)
            xy_coords.append([H_pos, V_pos])
            now = datetime.now()
            #dd/mm/YY H:M:S
            date = now.strftime("%M_%S_%f")
            xy_coords.append(date)
          
            #add something with time
            
            # blue vertical bar
            draw_rect(H_pos + H_offset - 2, 0, 4, SCREEN_HEIGHT, stroke_size=10, color = col_blue)
            draw_rect(0, V_pos + V_offset - 2, SCREEN_WIDTH, 4, stroke_size=10, color = col_blue)
            # diagnostics
            draw_text("HCOEF: " + str(np.round(SBD_model_h, 2)), (510, 50), color=col_green)
            draw_text("VCOEF: " + str(np.round(SBD_model_v, 2)), (510, 100), color=col_green)
            draw_text("SBD : " + str(np.round(SBD)), (510, 150), color=col_green)
            draw_text("HPOS: " + str(np.round(H_pos)), (510, 250), color=col_green)
            draw_text("VPOS: " + str(np.round(V_pos)), (510, 300), color=col_green)
            draw_text("XOFF: " + str(H_offset), (510, 350), color=col_green)
            draw_text("YOFF: " + str(V_offset), (510, 400), color=col_green)
        
        if STATE == "Saved":
            msg = "SBG.csv saved. Backspace for back. Return for new cycle."
           
            # Fixed UI elements
        draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .9), color=col_white)

        # update the screen to display the changes you made
        pg.display.update()


# splits a frame horizontally
def split_frame(Frame):
    height, width = Frame.shape
    F_left =  Frame[0:height, 0:int(width/2)]
    F_right = Frame[0:height, int(width/2):width]
    return F_left, F_right

# Computes the split-frame brightness diff
def calc_SBD(frame_1, frame_2):
    bright_left = np.mean(frame_1)
    bright_right = np.mean(frame_2)
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

# Predicts y position based on SBD and SBD coefficients
def predict_VPOS(sbd, coef):
    V_pos = coef[0] + sbd * coef[1]
    return V_pos

## Converts a CV2 framebuffer into Pygame image (surface!)
def frame_to_surf(frame, dim):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # convert BGR (CV2) to RGB (Pygame)
    img = np.rot90(img) # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf

def draw_text(text, dim,
              color = (255, 255, 255),
              center = False,
              font = Font):
    x, y = dim
    rendered_text = font.render(text, True, color)
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


##this is the welcome message
#def draw_welcome():
    #text_surface = font.render("Eye tracking", True, col_black, BACKGR_COL)
    #text_rectangle = text_surface.get_rect()
    #text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
    #screen.blit(text_surface, text_rectangle)
    #text_surface = font_small.render("Press Spacebar to continue", True, col_black, BACKGR_COL)
    #text_rectangle = text_surface.get_rect()
    #text_rectangle.center = (SCREEN_SIZE[0]/2.0,300)
    #screen.blit(text_surface, text_rectangle)
    
main()
