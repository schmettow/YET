## YETI 12: Multiple point callibration horizontal + vertical
## Machine learning using Keras ## Not working yet!






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
from sklearn import linear_model
import statsmodels.formula.api as smf
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

##### Preparations #####

# CV

log.basicConfig(filename='YET.log',level=log.INFO)
YET = cv2.VideoCapture(1) 
if YET.isOpened():
    width = int(YET.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(YET.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps =  YET.get(cv2.CAP_PROP_FPS) #FPS stands for frames per second. This property is used to get the frame rate of the video.
    dim = (width, height)
else:
    print('Unable to load camera.')
    exit()
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml") #same folder

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
pg.display.set_caption("YETI_3: Measuring vertical brightness distribution")
FONT = pg.font.Font('freesansbold.ttf',40)

SCREEN = pg.display.get_surface()
#SCREEN.fill(BACKGR_COL)

font = pg.font.Font(None, 60)


def main():
    
    ## Initial State
    STATE = "Prepare" # Stimulus, Measure
    DETECTED = False
    BACKGR_COL = col_black
    i=0
    image = []
   # Pimage = []
    label = []
   # x_test = []
   #3X_Stim = []
   # Bright_L = []
   # Bright_R = []
   # Bright_diff = []
   # Eyes = []
    D_data = {}

    ## FAST LOOP
    while True:
        pg.display.get_surface().fill(BACKGR_COL) 

        # Frame processing
        ret, Frame = YET.read(1) #reading every frame
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        F_gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY) #set color to grey
        Eyes = eyeCascade.detectMultiScale(
                F_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(200, 200)) #gives [[x, y, w, h]] of detected eyes
        # Frame conditionals
        if len(Eyes) > 0:
            DETECTED = True
            (x, y, w, h) = Eyes[0] 
            F_eye = F_gray[y:y+h,x:x+w] # stukje frame met detected eye (y tot en met y+h)
            eye_array = np.array(F_eye)
            #eye_array = np.expand_dims(eye_array, axis=0)

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
                YET.release() #releases software and hardware resources so it doesn't lead to errors
                pg.quit()
                sys.exit()
        

        # Automatic transitionals
        if STATE == "Prepare":
            X = random.uniform(0,1) * SCREEN_SIZE[0] #random punt van cirkel (waar je naar kijkt!)
            Y = random.uniform(0,1) * SCREEN_SIZE[1]
            circle_position = (X,Y)
            STATE = "Stimulus"
        elif STATE == "Measure":
            image_names = 'image'+str(i)+'.jpg'
            cv2.imwrite(image_names,F_eye) #make dataset of images
            
           # Pimage.append(eye_array)
            image.append(image_names)
            #image.append(F_eye)
            label.append(circle_position)
            i+=1
            
            STATE = "Prepare"
        elif STATE == "Save":
           # print(Pimage[0])
            D_data = {"image": image,"label": label}
            DF_data = pd.DataFrame(D_data) 
            print(DF_data)
            
            train_datagen = ImageDataGenerator(
                rescale=1 / 255.0,
                validation_split=0.25) #validation will have 25% of the total images
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=DF_data,
                x_col="image",
                y_col= "label",
                target_size=(90, 70),
                batch_size=1,
                class_mode="categorical", ##?
                subset='training',
                shuffle=False,
                seed=None
            )
            valid_generator = train_datagen.flow_from_dataframe(
                dataframe=DF_data,
                x_col="image",
                y_col= "label",
                target_size=(90, 70),
                batch_size=1,
                class_mode="categorical", ##?
                subset='validation',
                shuffle=False,
                seed=None
            )
            #batch = next(train_generator)
            #print(batch[0].shape)
            #test_img = batch[0][0]
            #print(test_img.shape)
            #plt.show(test_img)
            #train_generator[0,0].show
            #print(train_generator)
            #x_train, y_train = train_generator
            #print(x_train)
            #(x_train, y_train) = D_data.load_data()
            #print(x_train[0],y_train[0])
            #x_train = tf.keras.utils.normalize(x_train, axis=1)
            #y_train = tf.keras.utils.normalize(y_train, axis=1)
            #print(x_train[0],y_train[0])
            new_model = SBG_fit(train_generator)
            new_model.save('multiple_point_callibration.model')
           # plt.imshow(train[0],cmap=plt.cm.binary)
           # plt.show()
            
            #myfile =  open(CSV, mode = "w")
            #writer = csv.writer(myfile)
            #for i in range(0, len(X_Stim)):
            #    thisrow = [X_Stim[i], Bright_L[i], Bright_R[i], Bright_diff[i]] #per rij opgeslagen: stimulus locatie, brightness links, rechts, verschil
            #    writer.writerow(thisrow)
            #myfile.close()
            STATE = "Follow"
            print(STATE)
        
      #  elif STATE == "Follow":
            
          


        # Presentitionals
        if STATE == "Stimulus":
            BACKGR_COL = col_black
            if DETECTED:
                Img = frame_to_surf(F_eye, (90, 70)) #in DETECTED state, dataframe alleen linker bovenhoek getoond
                draw_circ(X, Y, 20) #cirkel op X locatie
            else:
                Img = frame_to_surf(F_gray, (900, 700)) #geen oog detected, dataframe groot beeld
            
            SCREEN.blit(Img,(50,50))
            
        elif STATE == "Follow":
            BACKGR_COL = col_black
            if DETECTED:
                Img = frame_to_surf(F_eye, (90, 70)) #in DETECTED state, dataframe alleen linker bovenhoek getoond
              #  x_test = [F_eye]
               # predictions = new_model.predict(x_test)
                predict_model = tf.keras.models.load_model('multiple_point_callibration.model')
                xy_pos = predict_model.predict(eye_array) #Formulas are not working
                print(xy_pos) #It doesn't reFnew the data 
                draw_circ(xy_pos, SCREEN_SIZE[1]/2, 40 ,  stroke_size=10, color=(0, 255, 255))
            else:
                Img = frame_to_surf(F_gray, (900, 700)) #geen oog detected, dataframe groot beeld
            
            SCREEN.blit(Img,(50,50))

#        draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .9), color=col_gray)
        # update the screen to display the changes you made
        pg.display.update()

## Estimates model coefficients 
def SBG_fit(train_generator): #Something like this, but I couldn't test it yet
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
    model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
    #model.fit_generator(generator=train_generator, epochs=3)
   # model.fit(frames, circl, epochs=3)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss)
    print(val_acc)
    return(model)



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