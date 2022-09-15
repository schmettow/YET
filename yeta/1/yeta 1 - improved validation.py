## YETI 14: Collecting 2-dim calibration data for quadrant SBG
## input = calibration point table
## Results = table with target coordinates and quadrant brightness

import time

YETA = 1
YETA_NAME = "Yeta" + str(YETA)
TITLE = "Yeta 1: Slideshow"
AUTHOR = "M Schmettow, N Bierhuizen, GOF5(2021)"
EYECASC = "haarcascade_eye.xml"
PART_ID = str(time.time())
RESULTS = "yeta2_" + PART_ID + ".csv"
SLIDE_TIME = 2 # Set this to a long value for a user-controlled slideshow

import sys
import os
import logging as log

# DS
import numpy as np
from sklearn import linear_model as lm
import csv
# CV
import cv2 as cv
# PG
import pygame as pg
from pygame.locals import *

##### Preparations #####

# CV
log.basicConfig(filename='YET.log', level=log.INFO)
YET = cv.VideoCapture(1)
if YET.isOpened():
    width = int(YET.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(YET.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = YET.get(cv.CAP_PROP_FPS)
    dim = (width, height)
else:
    print('Unable to load camera.')
    exit()

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
SCREEN_W = 1000
SCREEN_H = 800
SCREEN_SIZE = (SCREEN_W, SCREEN_H)

pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(YETA_NAME)
FONT = pg.font.Font('freesansbold.ttf', int(20 * min(SCREEN_SIZE) / 800))
Font = pg.font.Font('freesansbold.ttf', int(15 * min(SCREEN_SIZE) / 800))

SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 60)

picture_time = 5


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
    run = 0
    H_offset, V_offset = (0, 0)
    this_pos = (0, 0)

    Eyes = []
    OBS_cols = ("Part", "Obs", "time","x", "y", "Picture")
    #write_coef(RESULTS, flatten([OBS_cols, ["Picture"]]), "Creation")
    write_coef(RESULTS, OBS_cols, "Creation")
    CAL = np.zeros(shape=(0, 8))
    OBS = np.zeros(shape=(0, len(OBS_cols)))
    obs = 0

    picturetarget = np.concatenate(read_coef("Images/PictureInfo.csv", 1)).ravel() # Get the picture names
    picture_amount = len(picturetarget)
    picture_shown = 1
    imagepath = "Images/" + picturetarget[0]

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
                # F_eye = F_gray
                # w_eye, h_eye = (width, height)
        else:
            F_eye = F_gray[y_eye:y_eye + h_eye, x_eye:x_eye + w_eye]

        if STATE == "Validate1" or STATE == "Validate2":
            this_quad = np.array(quad_bright(F_eye))
            this_quad.shape = (1, 4)
            this_pos = M_0.predict(this_quad)[0, :]
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
                        print(STATE + str(active_target))

            elif STATE == "Target":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Measure"
                    print(STATE + str(active_target))
            elif STATE == "Validate1":
                if event.type == KEYDOWN and event.key == K_LEFT:
                        H_offset -= 5
                if event.type == KEYDOWN and event.key == K_RIGHT:
                        H_offset += 5
                if event.type == KEYDOWN and event.key == K_x: # press the X key to reset the offset
                        H_offset = 0
                if event.type == KEYDOWN and event.key == K_DOWN:
                        V_offset -= 5
                if event.type == KEYDOWN and event.key == K_UP:
                        V_offset += 5
                if event.type == KEYDOWN and event.key == K_y: # press the Y key to reset the offset
                        V_offset = 0
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Timer"
                elif event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target"
                    active_target = 0  # reset
                    run = run + 1
            elif STATE == "Validate2":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Timer"
                elif event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Quick"
            elif STATE == "Save":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Train"
                if event.type == KEYDOWN and event.key == K_RETURN:
                    STATE = "Save"
                    print(OBS)
                if event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target"
                    active_target = 0  # reset
                    run = run + 1
#           elif STATE == "Image":
#               if event.type == KEYDOWN and event.key == K_SPACE:
#                   STATE = "Train"
#               elif event.type == KEYDOWN and event.key == K_c:
#                   STATE = "Quick"
            elif STATE == "Quick":
#                if event.type == KEYUP and event.key == K_c:
                if event.type == KEYDOWN and event.key == K_c:
                    STATE = "Adjust"

        if event.type == QUIT:
            YET.release()
            pg.quit()
            sys.exit()

        # Automatic transitionals
        if STATE == "Measure":
            obs = obs + 1
            this_id = (obs, run)
            this_targ = tuple(np.array(targets[active_target][0:2]))
            print(np.shape(this_targ))
            this_bright = quad_bright(F_eye)
            this_obs = this_id + this_bright + this_targ
            CAL = np.vstack((CAL, this_obs))

            if (active_target + 1) < n_targets:
                active_target = active_target + 1
                STATE = "Target"
                print(STATE + str(active_target))
            else:
                print(CAL)
                STATE = "Save"

        if STATE == "Train":
            M_0 = train_QBG(CAL)
            STATE = "Validate1"
            
        if STATE == "Timer":
            t1 = time.time()
            STATE = "Image"

        if STATE == "Image":
            obs = obs + 1 # observation number
            this_id = (PART_ID, obs, time.time())
            this_bright = quad_bright(F_eye)
            this_quad = np.array(this_bright)
            this_quad.shape = (1, 4)
            this_pos = M_0.predict(this_quad)[0, :]
            this_obs = this_id
            t2 = time.time()
            elapsed_time = t2 - t1 # elapsed time since STATE == "Image" started
            if elapsed_time > picture_time: #presented longer then defined: trial is over
                if picture_shown < picture_amount: # check for the amount of picture shown
                    picture_shown += 1
                    imagepath = "Images/" + picturetarget[picture_shown-1] # image path for picture (where 0 is the first picture)
                    STATE = "Quick"     
                else:
                    t3 = time.time()
                    STATE = "Thank You"
                    #print(STATE)
            write_coef(RESULTS, flatten([this_obs, this_pos, picturetarget[picture_shown-1]]), "Data") # write the data (where 0 is the first picture)
                
        if STATE == "Adjust":
            this_bright = quad_bright(F_eye)
            this_quad = np.array(this_bright)
            this_quad.shape = (1, 4)
            this_pos = M_0.predict(this_quad)[0, :]
            H_offset = float(targets[4][0]) - float(this_pos[0])
            V_offset = float(targets[4][1]) - float(this_pos[1])
            STATE = "Validate2"
            
        # Show thank you screen
        if STATE == "Thank You":
            if time.time() - t3 > 3: #after 3sec
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
            msg = "Press Space while looking at the orange circle."
            draw_text(msg, (SCREEN_SIZE[0]* .1, SCREEN_SIZE[1] * .75), color=col_green)
        elif STATE == "Validate1":
            msg = "Use arrows to change the offset if necessary. Press 'X' or 'Y' to reset the offsets."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .7), color=col_green)
            msg = "Press Space to continue the experiment."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .75), color=col_green)
            msg = "Press Backspace to redo the calibration and reset the experiment."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .8), color=col_green)
            draw_rect(this_pos[0] + H_offset - 1, 0, 2, SCREEN_H, stroke_size=1, color=col_green)
            draw_rect(0, this_pos[1] + V_offset - 1, SCREEN_W, 2, stroke_size=1, color=col_green)
            # diagnostics
            draw_text("HPOS: " + str(np.round(this_pos[0])), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .15), color=col_green)
            draw_text("VPOS: " + str(np.round(this_pos[1])), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .2), color=col_green)
            draw_text("XOFF: " + str(H_offset), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .25), color=col_green)
            draw_text("YOFF: " + str(V_offset), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .3), color=col_green)
        elif STATE == "Validate2":
            msg = "Press Space to continue the experiment."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .75), color=col_green)
            msg = "Press Backspace to redo the centring."
            draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .8), color=col_green)
            draw_rect(this_pos[0] + H_offset - 1, 0, 2, SCREEN_H, stroke_size=1, color=col_green)
            draw_rect(0, this_pos[1] + V_offset - 1, SCREEN_W, 2, stroke_size=1, color=col_green)
            # diagnostics
            draw_text("HPOS: " + str(np.round(this_pos[0])), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .15), color=col_green)
            draw_text("VPOS: " + str(np.round(this_pos[1])), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .2), color=col_green)
            draw_text("XOFF: " + str(H_offset), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .25), color=col_green)
            draw_text("YOFF: " + str(V_offset), (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .3), color=col_green)
        elif STATE == "Image":
            img = pg.image.load(imagepath)
            img = pg.transform.smoothscale(img, SCREEN_SIZE)
            SCREEN.blit(img, (0, 0))
        elif STATE == "Quick":
            drawTargets(SCREEN, targets, active=4)
            msg = "CLick on C when looking at the orange circle."
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
    Quad = Obs[:, 2:6]
    Pos = Obs[:, 6:8]
    model = lm.LinearRegression()
    model.fit(Quad, Pos)
    return model


# Predicts position based on quad-split
def predict_pos(data, model):
    predictions = model.predict(data)
    return predictions


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
    text_rectangle.center = (SCREEN_SIZE[0] * 0.5, SCREEN_SIZE[1] * 0.4)
    SCREEN.blit(text_surface, text_rectangle)


main()
