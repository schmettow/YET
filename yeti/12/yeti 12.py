## YETI 12: Multiple point callibration horizontal + vertical
## Uses Keras for multiple regression 
## input = sgb_diff horizontal & vertical
## output = X,Y (location)

YETI = 12
YETI_NAME = "Yeti" + str(YETI)
TITLE = "Multiple point callibration with keras"
AUTHOR = "NM BIERHUIZEN"

import sys
import logging as log
import datetime as dt
from time import time
import random
# DS
import numpy as np
import pandas as pd

import csv
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
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

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
    Y_Stim = []
    Bright_T = []
    Bright_B = []
    Bright_L = []
    Bright_R = []
    Bright_V = []
    Bright_H = []
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
            X = random.uniform(0,1) * SCREEN_SIZE[0]
            Y = random.uniform(0,1) * SCREEN_SIZE[1]
            circle_position = (X,Y)
            STATE = "Stimulus"
        elif STATE == "Measure":
            F_top =  F_eye[0:int(h/2), 0:w]
            F_bot = F_eye[int(h/2):h, 0:w]
            bright_top = np.mean(F_top)
            bright_bot = np.mean(F_bot)
            F_left =  F_eye[0:h, 0:int(w/2)]
            F_right = F_eye[0:h, int(w/2):w]
            bright_left = np.mean(F_left)
            bright_right = np.mean(F_right)
            print(str(circle_position) + "--> Vertical: " + str(bright_top) + str(bright_bot)+ "--> Horizontal: " + str(bright_left) + str(bright_right))
            bright_vert = bright_top - bright_bot
            bright_hor = bright_left - bright_right
            X_Stim.append(X)
            Y_Stim.append(Y)
            Bright_T.append(bright_top)
            Bright_B.append(bright_bot)
            Bright_L.append(bright_left)
            Bright_R.append(bright_right)
            Bright_V.append(bright_vert)
            Bright_H.append(bright_hor)
            STATE = "Prepare"
        elif STATE == "Save":
            D_data = {"x_position": X_Stim,"y_position": Y_Stim, "Top": Bright_T,"Bottom": Bright_B, "Left": Bright_L,"Rigth": Bright_R, "Vertical": Bright_V,"Horizontal": Bright_H}
            DF_data = pd.DataFrame(D_data)
            print(DF_data)
            np_data = DF_data.to_numpy()
            print(np_data)

         #   training_data = DF_data[["Vertical", "Horizontal"]].values.reshape(-1,1).astype(np.float64)
         #   labels = DF_data[["x_position", "y_position"]].values.reshape(-1,1).astype(np.float64)
            
            model = keras_model_(np_data)
            
            #keras_model = tf.keras.Sequential([
            #    layers.Dense(units=1)
            #])

            #keras_model.compile(
            #    optimizer=tf.optimizers.Adam(learning_rate=0.1),
            #    loss='mean_absolute_error')
            #fitting =  keras_model.fit(
            #    labels,
            #    training_data,
            #    epochs=3)
            
            STATE = "Follow"
            print(STATE)


        # Presentitionals
        if STATE == "Stimulus":
            BACKGR_COL = col_black
            if DETECTED:
                Img = frame_to_surf(F_eye, (90, 70))
                draw_circ(X, Y, 20)
            else:
                Img = frame_to_surf(F_gray, (900, 700))
            
            SCREEN.blit(Img,(50,50))
            
            
        elif STATE == "Follow":
            BACKGR_COL = col_black
            if DETECTED:
                F_top =  F_eye[0:int(h/2), 0:w]
                F_bot = F_eye[int(h/2):h, 0:w]
                bright_top = np.mean(F_top)
                bright_bot = np.mean(F_bot)
                F_left =  F_eye[0:h, 0:int(w/2)]
                F_right = F_eye[0:h, int(w/2):w]
                bright_left = np.mean(F_left)
                bright_right = np.mean(F_right)
                bright_vert = bright_top - bright_bot
                bright_hor = bright_left - bright_right
                #test_list = [bright_vert, bright_hor]
                test_data = {"Vertical": bright_vert,"Horizontal": bright_hor}
                DF_test_data = pd.DataFrame(test_data, index=[0])
                print(DF_test_data)
                np_data = DF_test_data.to_numpy()
                new_pos = model.predict(np_data) 
                print(new_pos)
                new_df = pd.DataFrame(data=new_pos, columns=["X", "Y"])
                new_X = new_df["X"]
                new_Y = new_df["Y"]
                
                Img = frame_to_surf(F_eye, (90, 70)) #in DETECTED state, dataframe alleen linker bovenhoek getoond
                draw_circ(new_X, new_Y, 40 , stroke_size=10, color=(0, 255, 255))
                
            else:
                Img = frame_to_surf(F_gray, (900, 700)) #geen oog detected, dataframe groot beeld
            
            SCREEN.blit(Img,(50,50))
        
        # update the screen to display the changes you made
        pg.display.update()


## Standardise data
def stand_data(dataset):
    #NP_data = np.column_stack([X, Y, Top, Bottom, Left, Right, Vertical, Horizontal])
    A = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    return A

def keras_fit(dataset):

    keras_model = tf.keras.Sequential(x = dataset[:, 6:8], y = dataset[:, 0:2])
    return(keras_model)
    
    
def keras_model(dataset):
    model = Sequential()
    model.add(Dense(2, input_dim = 2, activation = 'linear')) #two input variables: diff_vert & diff_hor
    model.compile(loss = 'mse', optimizer = 'rmsprop', metrics=['mse'])
    model.fit(x = dataset[:, 6:8], y = dataset[:, 0:2], epochs = 3) # for x all rows but only first two collumns
    return model


def keras_model_(dataset):
    model = Sequential()
    model.add(Dense(2, input_dim = 2, activation = 'linear')) #two input variables: diff_vert & diff_hor
    model.compile(loss = 'mse', optimizer = 'rmsprop', metrics=['mse'])
    model.fit(x = dataset[:, 6:8], y = dataset[:, 0:2], epochs = 256, verbose=1) # for x all rows but only first two collumns
    model.fit(x = dataset[:, 6:8], y = dataset[:, 0:2], epochs = 256, verbose=1) #train again
    model.fit(x = dataset[:, 6:8], y = dataset[:, 0:2], epochs = 1024, verbose=0) # and again
    return model

    
def keras_predict(trained_model, dataset):
    predictions = trained_model.predict(dataset[:,6:8])
    result = predictions.T
    return result
    
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

## NORMALIZER UITZETTEN ZODAT X EN Y POSITIES ZIJN OP SCHERM IPV LINKSBOVEN
