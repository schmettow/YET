## YETI 13: Four point callibration horizontal + vertical
## Uses Keras for multiple regression 
## input = sgb_diff horizontal & vertical
## output = X,Y (location)

YETI = 13
YETI_NAME = "Yeti" + str(YETI)
TITLE = YETI_NAME + "Four point callibration with Keras"
AUTHOR = "NM BIERHUIZEN"

DEBUG = False

import sys
import logging as log
import datetime as dt
from time import time
import random
# DS
import numpy as np
import pandas as pd
# CV
import cv2
# PG
import pygame as pg
from pygame.locals import *
from pygame.compat import unichr_, unicode_
#keras
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
#print(tf.__version__)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


##### DEFINITIONS ####

## width and height in pixel
SCREEN_SIZE = (1400,600)
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]
XPOS  = (20, SCREEN_WIDTH - 20) ## x points for measuring
YPOS  = (20, SCREEN_HEIGHT - 20)
pos_U = (SCREEN_WIDTH/2, YPOS[0])
pos_D = (SCREEN_WIDTH/2, YPOS[1])
pos_L = (XPOS[0], SCREEN_HEIGHT/2)
pos_R = (XPOS[1], SCREEN_HEIGHT/2)


##### Preparations #####

# Reading the CV model for eye detection
eyeCascadeFile = "../trained_models/haarcascade_eye.xml"
if os.path.isfile(eyeCascadeFile):
    eyeCascade = cv2.CascadeClassifier(eyeCascadeFile)
else:
    sys.exit(eyeCascadeFile + ' not found. CWD: ' + os.getcwd())


# Connecting to YET
log.basicConfig(filename='YET.log',level=log.INFO)
YET = cv2.VideoCapture(1)
if not YET.isOpened():
        print('Unable to load camera.')
        exit()



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


# initialize PG window
pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(TITLE)

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
            # when eye is detected, set the trigger, store the rectangle and get the subframe
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
            if event.type == KEYDOWN:
                # Interactive transition conditionals (ITC)
                if STATE == "Detect":
                    if event.key == K_SPACE:
                        if Detected:
                            STATE = "Measure_U"
                elif STATE == "Measure_U":
                    if event.key == K_SPACE:
                        STATE = "Measure_D"
                        sbg_U_V = sbg_diff[0] #collecting up SBG_diff (vertical)
                        sbg_U_H = sbg_diff[1]
                    elif event.key == K_BACKSPACE:
                        STATE = "Detect"
                elif STATE == "Measure_D":
                    if event.key == K_SPACE:
                        STATE = "Measure_L"
                        sbg_D_V = sbg_diff[0] #collecting down SBG_diff (vertical)
                        sbg_D_H = sbg_diff[1]
                       # sbg_coef_ver = SBG_fit(sbg_U_V, sbg_D_V, YPOS[0], YPOS[1]) # fitting the model ;)
                    elif event.key == K_BACKSPACE:
                        STATE = "Detect"
                elif STATE == "Measure_L":
                    if event.key == K_SPACE:
                        STATE = "Measure_R"
                        sbg_L_V = sbg_diff[0]
                        sbg_L_H = sbg_diff[1] #collecting left SBG_diff
                    elif event.key == K_BACKSPACE:
                        STATE = "Detect"
                elif STATE == "Measure_R":
                    if event.key == K_SPACE:
                        STATE = "Save"
                        sbg_R_V = sbg_diff[0]
                        sbg_R_H = sbg_diff[1] #collecting right SBG_diff
                      #  sbg_coef_hor = SBG_fit(sbg_L_H, sbg_R_H, XPOS[0], XPOS[1]) # fitting the model
                    elif event.key == K_BACKSPACE:
                        STATE = "Detect"
                elif STATE == "Follow":
                    if event.key == K_BACKSPACE:
                        STATE = "Detect"
                print(STATE)

            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()
                


        # Automatic transitionals
        if STATE == "Save":                                     
            D_data = {"x_position": [pos_U[0], pos_D[0], pos_L[0], pos_R[0]] ,"y_position": [pos_U[1], pos_D[1], pos_L[1], pos_R[1]], "Vertical": [sbg_U_V, sbg_D_V, sbg_L_V, sbg_R_V], "Horizontal": [sbg_U_H, sbg_D_H, sbg_L_H, sbg_R_H]}
            DF_data = pd.DataFrame(D_data)
            print(DF_data)
            np_data = DF_data.to_numpy()
            print(np_data)
           
            model = keras_model(np_data)
           
            STATE = "Follow"
            print(STATE)
               
            

        # Presentitionals
        BACKGR_COL = col_white
        SCREEN.fill(BACKGR_COL)
        
        if Detected:
            Img = frame_to_surf(F_eye, (200, 200))
        else:
            Img = frame_to_surf(F_gray, (200, 200))

        if STATE == "Detect":
            BACKGR_COL = col_red_dim
            msg = "When eye is detected, press Space"

        if STATE == "Measure_U":
            msg = "Focus on the circle (up) and press Space. Backspace for back"
            draw_circ(SCREEN_WIDTH/2, YPOS[0], 20, stroke_size=5, color = col_black) #CHANGED FOR VERTICAL

        if STATE == "Measure_D":
            msg = "Focus on the circle (down) and press Space. Backspace for back"
            draw_circ(SCREEN_WIDTH/2, YPOS[1], 20, stroke_size=5, color = col_black) #CHANGED FOR VERTICAL

        if STATE == "Measure_L":
            msg = "Focus on the circle to your Left and press Space.  Backspace for back."
            draw_circ(XPOS[0], SCREEN_HEIGHT/2, 20, stroke_size=5, color = col_black)

        if STATE == "Measure_R":
            msg = "Focus on the circle to your Right and press Space. Backspace for back."
            draw_circ(XPOS[1], SCREEN_HEIGHT/2, 20, color = col_black, stroke_size=5)

        if STATE == "Follow":
            msg = "Check if the callibration has succeeded. Backspace for back."
            test_data = {"Vertical": sbg_diff[0],"Horizontal": sbg_diff[1]}
            DF_test_data = pd.DataFrame(test_data, index=[0])
            print(DF_test_data)
            np_data = DF_test_data.to_numpy()
            new_pos = model.predict(np_data) 
            print(new_pos)
            new_df = pd.DataFrame(data=new_pos, columns=["X", "Y"]) 
            new_X = new_df["X"]
            new_Y = new_df["Y"]
            draw_circ(new_X, new_Y, 40 ,  stroke_size=10, color=(0, 255, 255))
        
        
        # Fixed UI elements
        draw_text(msg, (SCREEN_WIDTH * .1, SCREEN_HEIGHT * .9), color=col_black)
        SCREEN.blit(Img,(50,50))
                    
        # update the screen to display the changes you made
        pg.display.update()


# Estimates the split-frame brightness diff
def SBG_diff(frame):
    height, width = frame.shape
    F_up =  frame[0:int(height/2), 0:width] 
    F_down = frame[int(height/2):height, 0:width] 
    bright_up = np.mean(F_up)
    bright_down = np.mean(F_down)
    bright_diff_ver = bright_up - bright_down
    
    F_left =  frame[0:height, 0:int(width/2)]
    F_right = frame[0:height, int(width/2):width]
    bright_left = np.mean(F_left)
    bright_right = np.mean(F_right)
    bright_diff_hor = bright_left - bright_right
    return bright_diff_ver, bright_diff_hor

def keras_model(dataset):
    model = Sequential()
    model.add(Dense(2, input_dim = 2, activation = 'linear')) #two input variables: diff_vert & diff_hor
    model.compile(loss = 'mse', optimizer = 'rmsprop', metrics=['mse'])
    model.fit(x = dataset[:, 2:4], y = dataset[:, 0:2], epochs = 256, verbose=1) # for x all rows but only first two collumns
    model.fit(x = dataset[:, 2:4], y = dataset[:, 0:2], epochs = 256, verbose=1) #train again
    model.fit(x = dataset[:, 2:4], y = dataset[:, 0:2], epochs = 1024, verbose=0) # and again
    return model

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
