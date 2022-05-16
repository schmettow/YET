## YETA 1 Slideshow with YET
## Input: Images and PictureInfo.csv file
## Results = table with Part, Picture, time, eye tracking coordinates

import sys
import os
import time

WD = os.path.dirname(sys.argv[0])
os.chdir(WD)

## Meta data of Yeta
YETA = 1
YETA_NAME = "Yeta" + str(YETA)
TITLE = "Yeta 1: Slideshow"
AUTHOR = "M Schmettow, N Bierhuizen, GOF5(2021)"
EYECASC = "haarcascade_eye.xml"

## Experiment
EXP_ID = "UV22"
EXPERIMENTER = "MS"
IMG_PATH = os.path.join(WD, "Images")
IMG_INFO = os.path.join(IMG_PATH, "PictureInfo.csv")
RESULT_DIR = "CSV"
PART_ID = str(int(time.time()))
RESULT_FILE = os.path.join(RESULT_DIR, "yeta1_" + EXP_ID + EXPERIMENTER + PART_ID + ".csv")
SCREEN_W = 1000
SCREEN_H = 1000
USB = 1 # Set the video device (typically 1, sometimes 0 or 2)
SLIDE_TIME = 1 # Set this to a long value for a user-controlled slideshow

import logging as log

# DS
import numpy as np
from sklearn import linear_model as lm
import pandas as pd
import csv


# CV
import cv2 as cv
# PG
import pygame as pg
from pygame.locals import *

##### Preparations #####

# CV
log.basicConfig(filename='YET.log', level=log.INFO)
YET = cv.VideoCapture(USB)
if YET.isOpened():
    width = int(YET.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(YET.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = YET.get(cv.CAP_PROP_FPS)
    dim = (width, height)
else:
    print('Unable to load camera.')
    sys.exit()

# Reading picture infos
if os.path.isfile(IMG_INFO):
    IMGS = pd.read_csv(IMG_INFO)
    print(IMGS)
else:
    print(IMG_INFO  + ' not found. CWD: ' + os.getcwd())
    sys.exit()


# Reading the CV model for eye detection
if os.path.isfile(EYECASC):
    eyeCascade = cv.CascadeClassifier(EYECASC)
else:
    sys.exit(EYECASC  + ' not found. CWD: ' + os.getcwd())

# Colour definement
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
    ## Initial State
    STATE = "Detect"  # Measure, Target
    DETECTED = False
    BACKGR_COL = col_black
    
    CALIBRATIONPERCENTAGE = np.array([0.125, 0.5, 0.875])
    CALIBRATIONPOINTS = np.multiply([[SCREEN_SIZE[1]], [SCREEN_SIZE[0]]], CALIBRATIONPERCENTAGE) # x and y positions
    CALIBRATIONPOINTS = np.round(CALIBRATIONPOINTS) # round the value to the nearest integer value
    CALIBRATIONPOINTS = CALIBRATIONPOINTS.astype(int) # make it an integer
    #print(CALIBRATIONPOINTS)
    
    targets = np.empty([1, 2]) # create an array of correct dimensions for the targets
    for i in range(len(CALIBRATIONPOINTS[1])):
        values_y = CALIBRATIONPOINTS[0, i] * np.ones([len(CALIBRATIONPOINTS[0]), 1]) # Make a y value array
        values_x = np.transpose(CALIBRATIONPOINTS[[1], :]) # make a x value array
        values_x = np.append(values_x, values_y, axis=1) # combine arrays
        targets = np.append(targets, values_x, axis=0) # put into the targets array
    targets = np.delete(targets, 0, axis=0)
    #print(targets)
    
    n_targets = len(targets) # How many calibrationpoints are there
    active_target = 0
    H_offset, V_offset = (0, 0)
    this_pos = (0, 0)

    Eyes = []

    OBS_cols = ("Part", "Picture", "time","xraw", "yraw", "x", "y") 
    OBS = pd.DataFrame(columns = OBS_cols, dtype = "float64")
    OBS["Picture"].astype("category")
    obs = 0
    CAL = np.zeros(shape=(0, 6))

    picturetarget = IMGS["File"]
    picture_amount = len(picturetarget)
    picture_shown = 1
    this_image = picturetarget[0]
    this_path = os.path.join(IMG_PATH, this_image)

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
                F_eye = F_gray[y_eye:y_eye + h_eye, x_eye:x_eye + w_eye]
            else:
                DETECTED = False
        # in all other states, the eye coordinates are updated
        else:
            F_eye = F_gray[y_eye:y_eye + h_eye, x_eye:x_eye + w_eye]

        if STATE == "Validate":
            this_pos = predict_pos(F_eye, M_0)
            # print(this_pos)

        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Detect":
                if DETECTED:
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        this_Eye = Eyes[0]
                        x_eye, y_eye, w_eye, h_eye = this_Eye
                        STATE = "Target"
            elif STATE == "Target":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Measure"
            elif STATE == "Validate":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "prepareImage"
                elif event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target"
                    active_target = 0  # reset
            elif STATE == "Save":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Train"
                if event.type == KEYDOWN and event.key == K_RETURN:
                    STATE = "Save"
                if event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target"
                    active_target = 0  # reset
            # elif STATE == "Image":
            #    if event.type == KEYDOWN and event.key == K_SPACE:
            #        STATE = "Quick"
            elif STATE == "Quick":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    eyex, eyey = predict_pos(F_eye, M_0)
                    H_offset = float(targets[4][0]) - float(eyex)
                    V_offset = float(targets[4][1]) - float(eyey)
                    STATE = "prepareImage"

            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()


        # Automatic transitionals
        if STATE == "Measure":
            obs = obs + 1
            targetx, targety = np.array(targets[active_target][0:2])
            this_measure = quad_bright(F_eye)
            this_obs = this_measure + (targetx, targety) # combines arrays
            CAL = np.vstack((CAL, this_obs))
            if (active_target + 1) < n_targets:
                active_target = active_target + 1
                STATE = "Target"
                print(STATE + str(active_target))
            else:
                STATE = "Save"
                print(STATE)

        if STATE == "Train":
            M_0 = train_QBG(CAL)
            STATE = "Validate"
            print(STATE)

        if STATE == "prepareImage":
            picture_shown += 1
            this_image = IMGS["File"][picture_shown - 1]
            this_path = os.path.join(IMG_PATH, this_image)
            IMG = pg.image.load(this_path)
            this_xscale = SCREEN_W/IMG.get_width()
            this_yscale = SCREEN_H/IMG.get_height()
            IMG = pg.transform.smoothscale(IMG, SCREEN_SIZE)
            t_image_started = time.time()
            STATE = "Image"
            
        if STATE == "Image":
            obs = obs + 1 # observation number
            eyex, eyey = predict_pos(F_eye, M_0)
            eyex = eyex + H_offset
            eyey = eyey + V_offset
            this_pos = (eyex, eyey)
            elapsed_time = time.time() - t_image_started # elapsed time since STATE == "Image" started
            if elapsed_time > SLIDE_TIME: #presented longer then defined: trial is over
                OBS.to_csv(RESULT_FILE, index = False) ## auto save results after every slide
                if picture_shown < picture_amount: # check for the amount of picture shown
                    STATE = "Quick"
                    print(STATE)    
                else:
                    t3 = time.time()
                    STATE = "Thank You"
                    print(STATE)
            this_OBS = pd.DataFrame({"Part": PART_ID, "Picture": this_image, 
                                    "time" : time.time(), 
                                    "xraw": eyex, "yraw": eyey, 
                                    "x": eyex/this_xscale, "y": eyey/this_yscale}, index = [0])
            OBS = pd.concat([OBS, this_OBS])
                
            
        # Show thank you screen
        if STATE == "Thank You":
            if time.time() - t3 > 3: #after 3sec
                STATE = "Exit"
                print(STATE)
                OBS.to_csv(RESULT_FILE, index = False)
                print("Saved to " + RESULT_FILE)
                YET.release()
                pg.quit()
                sys.exit()

        # Presentitionals
        # over-paint previous with background xcolor
        pg.display.get_surface().fill(BACKGR_COL)

        if STATE == "Detect":
            if DETECTED:
                Img = frame_to_surf(F_eye, (int(SCREEN_SIZE[0] * .5), int(SCREEN_SIZE[1] * .5)))
                msg = "Eye detected press Space to continue."
                draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .85), color=col_green)
            else:
                Img = frame_to_surf(Frame, (int(SCREEN_SIZE[0] * .5), int(SCREEN_SIZE[1] * .5)))
                msg = "Trying to detect an eye."
                draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .85), color=col_green)
            SCREEN.blit(Img, (int(SCREEN_SIZE[0] * .25), int(SCREEN_SIZE[1] * .25)))
        elif STATE == "Save":
            msg = "Press Space to continue."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .45), color=col_green)
            msg = "Press Backspace to redo the calibration."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .55), color=col_green)
        elif STATE == "Target":
            if DETECTED:
                drawTargets(SCREEN, targets, active_target)
            msg = "Follow the orange light and press Space."
            draw_text(msg, (SCREEN_SIZE[0]* .1, SCREEN_SIZE[1] * .75), color=col_green)
        elif STATE == "Validate":
            msg = "Press Space to continue the experiment."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .75), color=col_green)
            msg = "Press Backspace to redo the calibration and reset the experiment."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .8), color=col_green)
            draw_rect(this_pos[0] + H_offset, 0, 2, SCREEN_H, stroke_size=1, color=col_green)
            draw_rect(0, this_pos[1] + V_offset, SCREEN_W, 2, stroke_size=1, color=col_green)
            # diagnostics

            draw_text("HPOS: " + str(np.round(this_pos[0])), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .15), color=col_green)
            draw_text("VPOS: " + str(np.round(this_pos[1])), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .2), color=col_green)
            draw_text("XOFF: " + str(H_offset), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .25), color=col_green)
            draw_text("YOFF: " + str(V_offset), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .3), color=col_green)
        elif STATE == "Image":
            SCREEN.blit(IMG, (0, 0))
        elif STATE == "Quick":
            drawTargets(SCREEN, targets, active=4)
            msg = "Look at the orange circle and press Space."
            draw_text(msg, (SCREEN_SIZE[0] * .05, SCREEN_SIZE[1] * .75), color=col_green)
        elif STATE == "Thank You":
            draw_thank_you()
        # update the screen to display the changes you made
        pg.display.update()


def drawTargets(screen, targets, active=0):
    cnt = 0
    for target in targets:
        pos = list(map(int, target))
        color = (160, 160, 160)
        radius = 20
        stroke = 10
        if cnt == active: color = (255, 120, 0)
        cnt += 1
        pg.draw.circle(screen, color, pos, radius, stroke)


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


def quad_bright(frame):
    w, h = np.shape(frame)
    b_NW = np.mean(frame[0:int(h / 2), 0:int(w / 2)])
    b_NE = np.mean(frame[int(h / 2):h, 0:int(w / 2)])
    b_SW = np.mean(frame[0:int(h / 2), int(w / 2):w])
    b_SE = np.mean(frame[int(h / 2):h, int(w / 2):w])
    out = (b_NW, b_NE, b_SW, b_SE)
    # out = np.array((b_NW, b_NE, b_SW, b_SE))
    # out.shape = (1,4)
    return (out)


def train_QBG(Obs):
    Quad = Obs[:, 0:4]
    Pos = Obs[:, 4:6]
    model = lm.LinearRegression()
    model.fit(Quad, Pos)
    return model


# Predicts position based on quad-split
def predict_pos(frame, model):
    bright = quad_bright(frame)
    quad = np.array(bright)
    quad.shape = (1, 4)
    eyex, eyey = model.predict(quad)[0, :]
    return eyex, eyey



def write_coef(filename, text, option):
    if option == "Creation":
        with open(filename, "w") as csvfile:  # write in Data.csv and append the new info to the bottom
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)  # library usages
            writer.writerow(text)  # write the text
    elif option == "Data":
        with open(filename, "a", newline='\n') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(text)


def read_coef(filename, dim):
    coordinates = np.arange(dim)
    with open(filename, newline='\n') as csvfile:
        readText = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in readText:
            coordinates = np.vstack((coordinates, row))
        coordinates = np.delete(coordinates, [0, 1], axis=0)
    return coordinates


# Unnest nested lists
def flatten(text):
    return [item for sublist in text for item in sublist]


def draw_thank_you():
    text_surface = FONT.render("Thank You", True, col_white, col_black)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] * 0.5, SCREEN_SIZE[1] * 0.3)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = FONT.render("For participation", True, col_white, col_black)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] * 0.5, SCREEN_SIZE[1] * 0.5)
    SCREEN.blit(text_surface, text_rectangle)


main()
