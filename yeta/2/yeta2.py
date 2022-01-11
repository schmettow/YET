# Yeta2: Stroop experiment
# Running the Stroop task with eye tracking

from time import time

YETA = 2
YETI = 8
YETI_NAME = "Yeta" + str(YETI)
TITLE = "Stroop task"
AUTHOR = "M SCHMETTOW, GOF17 (2021)"
#CONFIG = ""
EYECASC = "../../trained_models/haarcascade_eye.xml"
RESULTS = "14/yeti_14_" + str(time()) + ".csv"
DEBUG = True #17: can be changed to false for the experiment
MOUSE = True  #17: Changed to false because we now use the eyes instead of the mouse
USB_YET = 1


# -*- coding: utf-8 -*-
import os
import csv
import datetime
import random
import sys
from time import time

# CV
import cv2 as cv
# DS
import numpy as np
# PG
import pygame as pg
from pygame.locals import *


##### DEFINITIONS ####
## width and height in pixel

SCREEN_SIZE = (1400, 500)
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]
HPOS = (40, SCREEN_WIDTH - 40)   # x points for measuring
VPOS = (40, SCREEN_HEIGHT - 40)  # y pointes for measuring

##### Preparations #####
# Reading the CV model for eye detection
if os.path.isfile(EYECASC):
    eyeCascade = cv.CascadeClassifier(EYECASC)
else:
    sys.exit(EYECASC + ' not found. CWD: ' + os.getcwd())

# color definitions

alpha = 30 #17: add alpha to a colour to make it see-through, so that we can have texts in boxes

col_black = (0, 0, 0)
col_red = (250, 0, 0)
col_red_alpha = (250, 0, 0, alpha)
col_green = (0, 255, 0)
col_green_alpha = (0, 255, 0, alpha)
col_blue = (0, 0, 255)
col_blue_alpha = (0, 0, 255, alpha)
col_yellow = (250, 250, 0)
col_white = (255, 255, 255)
col_red_dim = (120, 0, 0)
col_green_dim = (0, 60, 0)
col_yellow_dim = (120, 120, 0)
col_gray = (220, 220, 220)
col_gray_alpha = (220, 220, 220, alpha)
col_orange = (250, 140, 0)
col_orange_alpha = (250, 140, 0, 128)

#Make a hit dictionary to determine if detection in a box has occurred for x amount of time
hit = {"red": 0.0, "green": 0.0, "blue": 0.0} #17: to safe time being spent with the eye in the answer boxes

#Make a timer value after which the answer is accepted, in seconds
accepted_hit_timer = .1 #17: the number of seconds in which looking at the answer is accepted as the actual answer

#Make a fail timer
fail_timer = 3 #17: same as above but for not answering

#Make a filename, a csv header and list to store results in
details = ["Participant", "Trail_nr", "Correct", "Reaction time"] #17: colomn names (dont change the order)
results = []

# Experiment
n_trials = 6

WORDS = ("red", "green", "blue", "orange")
COLORS = {"red": col_red,
          "green": col_green,
          "blue": col_blue,
          "orange": col_orange}

KEYS = {"red": K_b,
        "green": K_n,
        "blue": K_m,
        "orange": K_o}
BACKGR_COL = col_gray

#########################################################

# initialize Pygame
pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(TITLE)
Font = pg.font.Font('freesansbold.ttf', 30)
Font_small = pg.font.Font('freesansbold.ttf', 15)
SCREEN = pg.display.get_surface()
SCREEN_OVERLAY = pg.display.get_surface()

font = pg.font.Font(None, 80) #from Stroop task
font_small = pg.font.Font(None, 40) #from Stroop task



def main():
    #for Stroop task
    participant = "Yumi" #EXPERIMENT: change for participants name
    trial_number = 0
    this_reaction_time = 0
    time_when_presented = 0
    this_correctness = False
    this_word = ''

    ##for eye-tracker - Initial interaction State

    STATE = "welcome"  # PROJECT: add this so that welcome is the first screen
    Detected = False  # eye detection
    Eyes = []  # results of eye detection
    SBD_H = 0  # split frame difference for horizontal
    SBD_V = 0  # split frame difference for vertical
    H_offset = 0  # manual horizontal offset
    V_offset = 0  # manual vertical offset

    # Connecting to YET
    YET = cv.VideoCapture(USB_YET) #17: number defines what camera is being used

    ###################################


    if YET.isOpened():
        width = int(YET.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(YET.get(cv.CAP_PROP_FRAME_HEIGHT))
        print(width, height)
        fps = YET.get(cv.CAP_PROP_FPS)
        dim = (width, height)
        print('YET stream' + str(dim) + "@" + str(fps)) #17: can remove this

    else:
        print('Unable to load YET.')
        exit(0)

    ## FAST LOOP

    while True:

        # General Frame Processing
        ret, Frame = YET.read(1)

        msg = "" #17: every loop the message is deleted again so that there are no old messages in the screen

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # collecting a grayscale frame
        F_gray = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)

        # Conditional Frame Processing
        if STATE == "Detect":

            # using the eye detection model on the frame
            Eyes = eyeCascade.detectMultiScale(F_gray, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))

            # when eye is detected, set the trigger, store the rectangle and get the subframe
            if len(Eyes) > 0:
                Detected = True
                (x, y, w, h) = Eyes[0]
                F_eye = F_gray[y:y + h, x:x + w]

        # in all other states, we only extract teh eye frane and compute SBG_diff

        elif STATE != "welcome":  #add this so that the program knows what to do when the state is not 'welcome'
            F_eye = F_gray[y:y + h, x:x + w]  #17: redefine the eye coordinates
            F_left, F_right, F_top, F_bottom = split_frame(F_eye)  # complete split frame
            SBD_H = calc_SBD(F_left, F_right)  # SBD for horizontal
            SBD_V = calc_SBD(F_top, F_bottom)  # SBD for vertical

        ## Event handling
        for event in pg.event.get():
            if event.type == KEYDOWN:

                # Interactive transition conditionals (ITC)
                ## Five consecutive steps with back-and-forth navigation
                if STATE == "welcome":  # PROJECT: add this with 'if' so that it is the first screen
                    if event.key == K_SPACE:  # PROJECT: add this so you can continue to the next screen with spacebar
                        STATE = "Detect"  # PROJECT: add this so 'detect' is the next screen if spacebar is pressed

                elif STATE == "Detect":  # PROJECT: this was first 'if', but since we added the welcome screen this is changed to 'elif'
                    if event.key == K_SPACE:
                        if Detected:
                            STATE = "Measure_L"
                    elif event.key == K_BACKSPACE:  # addition
                        STATE = "welcome"

                elif STATE == "Measure_L":
                    if event.key == K_SPACE:
                        SBD_L = SBD_H  # collecting left SBG_diff fro horizontal
                        if DEBUG: print(SBD_L)
                        STATE = "Measure_R"
                    elif event.key == K_BACKSPACE:
                        STATE = "Detect"

                elif STATE == "Measure_R":
                    if event.key == K_SPACE:
                        SBD_R = SBD_H  # collecting right SBG_diff for horizontal
                        STATE = "Measure_T"  # Automkatic Transitional #added for vertical
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_L"

                elif STATE == "Measure_T":
                    if event.key == K_SPACE:
                        SBD_T = SBD_V  # collecting top SBG_diff for vertical
                        if DEBUG: print(SBD_T)
                        STATE = "Measure_B"
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_R"

                elif STATE == "Measure_B":
                    if event.key == K_SPACE:
                        SBD_B = SBD_V  # collecting bottom SBG_diff for vertical
                        STATE = "Train"  # Automkatic Transitional
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_T"

                elif STATE == "Validate":
                    if event.key == K_LEFT:  # event key for left_H
                        H_offset -= 5
                    if event.key == K_RIGHT:  # event key for right_H
                        H_offset += 5
                    if event.key == K_DOWN:
                        H_offset = 0
                    if event.key == K_a:  # event key for top_V 
                        V_offset -= 5
                    if event.key == K_b:  # evenet key for bottom_V 
                        V_offset += 5
                    if event.key == K_c:  
                        V_offset = 0
                    # if event.key == K_SPACE:
                    #     write_csv(SBD_model_H, SBD_model_V)  # save coefficients for horizontal and vertical
                    #     STATE = "Save"
                    elif event.key == K_BACKSPACE:
                        STATE = "Measure_B"

                    #############################################################
                    #17: to combine the stroop task with the eye tracker 
                    elif event.key == K_RETURN:
                        STATE = "Stroop_init"

                elif STATE == "Stroop_init":
                    if event.key == K_BACKSPACE:
                        STATE = "Validate"
                    if event.key == K_SPACE:
                        STATE = "Stroop_prep_trial"

                elif STATE == "Stroop_prep_trial":
                    if event.key == K_BACKSPACE:
                        STATE = "Stroop_init"
                    if event.key == K_SPACE:
                        STATE = "Stroop_trial"

                elif STATE == "Stroop_trial":
                    if event.key == K_BACKSPACE:
                        STATE = "Stroop_init"
                    elif event.type == KEYDOWN and event.key in KEYS.values():

                        time_when_reacted = time()
                        this_reaction_time = time_when_reacted - time_when_presented - accepted_hit_timer #17: add 'accepted_hit_timer' to make sure that the 1 second for answering is not counted

                        this_correctness = (event.key == KEYS[this_color])

                        # Append result to results list
                        results.append([participant, trial_number, this_correctness, this_reaction_time])
                        if DEBUG:
                            print(f"DEBUG results: {results}")
                        STATE = "Stroop_feedback"

                elif STATE == "Stroop_failed": #17: for when the 10 seconds answering time has passed
                    results.append(["Fail timer passed"])
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        if trial_number < n_trials:
                            STATE = "Stroop_prep_trial"
                        else:
                            STATE = "Stroop_goodbye"

                elif STATE == "Stroop_feedback":
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        if trial_number < n_trials:
                            STATE = "Stroop_prep_trial"
                        else:
                            STATE = "Stroop_goodbye"

                elif STATE == "Stroop_goodbye":
                    if event.key == K_BACKSPACE: #17: to restart the test
                        STATE = "Stroop_init"
                    elif event.key == K_s:
                        # Save results to CSV file
                        with open(RESULTS, 'a', newline='') as f: #17: 'a' is for append to an existing file, so that all data is in the same file
                            write = csv.writer(f)
                            write.writerow([datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")])
                            write.writerow(details)
                            write.writerows(results)
                            results.clear()
                        STATE = "Stroop_save"   
                ##############################################################

                elif STATE == "Stroop_save":
                    if event.key == K_BACKSPACE:
                        STATE = "Stroop_init" 
                    elif event.key == K_RETURN:
                        STATE = "Detect"  ## ring closed
                # print(STATE)

            if event.type == QUIT:
                YET.release()
                pg.quit()
                sys.exit()

        # Automatic transitionals

        if STATE == "Train":
            SBD_model_H = train_SBD((SBD_L, SBD_R), HPOS)  # fitting the model for L&R_H
            SBD_model_V = train_SBD((SBD_T, SBD_B), VPOS)  # fitting the model for T&R_V
            print("SBD_H: " + str((SBD_L, SBD_R)) + "Model: " + str(SBD_model_H))  # printing model SBD_MODEL H
            print("SBD_V: " + str((SBD_T, SBD_B)) + "Model: " + str(SBD_model_V))  # printing model SBD_MODEL V
            STATE = "Validate"
            print(STATE)

        # Presentitionals
        BACKGR_COL = col_black
        SCREEN.fill(BACKGR_COL)
        if STATE == "welcome":  # PROJECT: add this so that the computer knows what to do at the welcome screen
            draw_welcome()

        # YET stream
        if Detected:
            Img = frame_to_surf(F_eye, (400, 400))
        else:
            Img = frame_to_surf(F_gray, (400, 400))

        if STATE != "welcome":  # PROJECT: ! means 'not', so in this case 'if state is not welcome ...' 
            #####################################################
            #removes the eye on validation and strooptest
            do_not_draw_eye_frame_states = ["Validate", "Stroop_init", "Stroop_prep_trial", "Stroop_trial",
                                            "Stroop_feedback", "Stroop_goodbye", "Stroop_failed", "Stroop_save"]
            if STATE not in do_not_draw_eye_frame_states:
                SCREEN.blit(Img, ((SCREEN_WIDTH - 400) / 2, (SCREEN_HEIGHT - 400) / 2))

        if STATE == "welcome":  # PROJECT: add the text for the welcome screen
            msg = "Press spacebar to continue"
            draw_text("Welcome to the eye-tracking experiment of group 17", (340, 250), color=col_white)

        if STATE == "Detect":
            if Detected:
                msg = "Please Press Space"
            else:
                msg = "Eye not detected"

        if STATE == "Measure_L":  # Plot left circle
            msg = "Focus on the circle to your left and press Space.  Backspace for back."
            draw_circ(HPOS[0], SCREEN_HEIGHT / 2, 20, stroke_size=10, color=col_red)

        if STATE == "Measure_R":  # plot right circle
            msg = "Focus on the circle to your right and press Space. Backspace for back."
            draw_circ(HPOS[1], SCREEN_HEIGHT / 2, 20, color=col_red, stroke_size=10)

        if STATE == "Measure_T":  # plot top circle
            msg = "Focus on the circle at the top and press Space.  Backspace for back."
            draw_circ(700, 25, 20, stroke_size=10, color=col_red)

        if STATE == "Measure_B":  # plot bottom circle
            msg = "Focus on the circle at the bottom and press Space. Backspace for back."
            draw_circ(700, 475, 20, color=col_red, stroke_size=10)

        if STATE == "Validate":
            msg = "Press Space for saving.  Backspace for back.  Enter for stroop test."
            H_POS = predict_HPOS(SBD_H, SBD_model_H)  # predict HPOS
            V_POS = predict_VPOS(SBD_V, SBD_model_V)  # predict VPOS

            draw_lines(H_POS, H_offset, V_POS, V_offset)

            # diagnostics
            draw_text("COEF_H: " + str(np.round(SBD_model_H, 2)), (510, 50), color=col_green)  # add text for coef_H
            draw_text("COEF_V: " + str(np.round(SBD_model_V, 2)), (510, 100), color=col_green)  # add text for coef_V
            draw_text("SBD_H : " + str(np.round(SBD_H)), (510, 150), color=col_green)  # add text for sbd_H
            draw_text("SBD_V : " + str(np.round(SBD_V)), (510, 200), color=col_green)  # add text for sbd_v
            draw_text("HPOS: " + str(np.round(H_POS)), (510, 250), color=col_green)  # add text for hpos
            draw_text("VPOS: " + str(np.round(V_POS)), (510, 300), color=col_green)  # add text for vpos
            draw_text("XOFF: " + str(H_offset), (510, 350), color=col_green)  # add text for xoff
            draw_text("YOFF: " + str(V_offset), (510, 400), color=col_green)  # add text for yoff

        if "Stroop" in STATE:
            BACKGR_COL = col_gray
            SCREEN.fill(BACKGR_COL)

        if STATE == "Stroop_init":
            msg = ""
            trial_number = 0  # Reset the trail_number to zero
            draw_stroop_welcome()

        if STATE == "Stroop_prep_trial":
            # New trail so reset the previously saved timing(detection) values
            reset_saved_hit_values(reset_no_matter_what=True)

            trial_number += 1
            this_word = pick_color()
            this_color = pick_color()
            time_when_presented = time()
            STATE = "Stroop_trial"

        if STATE == "Stroop_trial":
            #################################################################
            #17: this shows the blue lines during the stroop task
            H_POS = predict_HPOS(SBD_H, SBD_model_H)  # predict HPOS
            V_POS = predict_VPOS(SBD_V, SBD_model_V)  # predict VPOS
            draw_lines(H_POS, H_offset, V_POS, V_offset)
            draw_stimulus(this_color, this_word)

            zones = draw_detection_zones_and_buttons()
            #17: red = zone 0, green = zone 1, blue = zone 2

            #variable name for eye location
            point = (H_POS, V_POS) #17: position of the blue cross, to detect where the eye is looking at for the answering

            if check_answer_within_acceptable_timeframe(time_when_presented): #17: subtracting the current time to the saved time from the start and see if it is greater than 10 seconds
                STATE = "Stroop_failed"

            check_point_within_zones(zones, point) #17: to see if the eye is looking in one of the answer boxes

        if STATE == "Stroop_failed":
            draw_failed()

        if STATE == "Stroop_feedback":
            draw_feedback(this_correctness, this_reaction_time)

        if STATE == "Stroop_goodbye":
            draw_goodbye()

        if STATE == "Stroop_save":
            msg = f"{RESULTS} saved. Backspace for back. Return for new cycle."

        # Fixed UI elements
        draw_text(msg, (SCREEN_SIZE[0] * .1, SCREEN_SIZE[1] * .9), color=col_white)

        # update the screen to display the changes you made
        pg.display.update()


def draw_lines(H_POS, H_offset, V_POS, V_offset): #17: made a function of this cross to make sure that the boxes can be used in the Stroop task as well
    draw_rect(H_POS + H_offset - 2, 0, 4, SCREEN_HEIGHT, stroke_size=1, color=col_blue)  # plot blue horizontal line
    draw_rect(0, V_POS + V_offset - 2, SCREEN_WIDTH, 4, stroke_size=1, color=col_blue)  # Plot blue vertical line


# splits a frame horizontally
def draw_welcome():  # PROJECT: add this so the computer has visuals for the welcome screen
    text_surface = Font.render("welcome", True, col_gray, col_gray)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 150)
    text_surface = Font_small.render("Press Spacebar to continue", True, col_gray, col_gray)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 300)


def split_frame(Frame):
    height, width = Frame.shape
    F_left = Frame[0:height, 0:int(width / 2)]
    F_right = Frame[0:height, int(width / 2):width]
    F_top = Frame[0:int(height / 2), 0:width]  # add frame for F-top
    F_bottom = Frame[int(height / 2):height, 0:width]  # add frame f-bottom
    return F_left, F_right, F_top, F_bottom  # close up def of split frame


# Computes the split-frame brightness diff

def calc_SBD(frame_1, frame_2):
    bright_left = np.mean(frame_1)
    bright_right = np.mean(frame_2)
    sbd = bright_right - bright_left
    return sbd


# Estimates linear coefficients from two points and their brightness diff #whole thing for project

def train_SBD(SBD, X):  # note: no changes is needed
    beta_1 = (X[1] - X[0]) / (SBD[1] - SBD[0])
    beta_0 = X[0] - SBD[0] * beta_1
    return (beta_0, beta_1)


# Predicts x position based on SBD and SBD coefficients

def predict_HPOS(SBD_H, coef):  # def predict of hpos
    H_POS = coef[0] + SBD_H * coef[1]
    return H_POS


def predict_VPOS(SBD_V, coef):  # def predict of vpos
    V_POS = coef[0] + SBD_V * coef[1]
    return V_POS


## Converts a cv framebuffer into Pygame image (surface!)

def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img)  # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


def draw_text(text, dim, color=(255, 255, 255), center=False, font=Font):
    x, y = dim
    rendered_text = font.render(text, True, color)

    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()

    # this sets the x and why coordinates
    if center:
        box.center = (x, y)
    else:
        box.topleft = (x, y)

    # This puts a pre-rendered object to the screen
    SCREEN.blit(rendered_text, box)


def draw_circ(x, y, radius, color=(255, 255, 255), stroke_size=1):
    pg.draw.circle(SCREEN, color, (x, y), radius, stroke_size)


def draw_rect(x, y, width, height, color=(255, 255, 255), stroke_size=1):
    pg.draw.rect(SCREEN, color, (x, y, width, height), stroke_size)


def draw_rect_alpha(surface, color, rect): #17: to draw rectangles but now with alpha (transparant), we already have the rectangles and now we add another rectangle that is transparant over the other rectangle
    shape_surf = pg.Surface(pg.Rect(rect).size, pg.SRCALPHA)
    pg.draw.rect(shape_surf, color, shape_surf.get_rect())
    surface.blit(shape_surf, rect)


#17: add this so that the rectangle is created first, and then drawn in this code. Instead of having to write coordinates in this code
def draw_rect_by_tuple(rect, color=(255, 255, 255), stroke_size=1):
    pg.draw.rect(SCREEN, color, rect, stroke_size)


# Function definitions
def pick_color():
    random_number = random.randint(0, 2)
    return WORDS[random_number]


# Added
def draw_detection_zones_and_buttons():
    stroke = 2 #17: thickness of the line
    width = SCREEN_WIDTH / 3
    height = SCREEN_HEIGHT / 3

    r1 = Rect(0, height * 2, width, height)  # height times 2 is two-third of the screen down
    c1 = r1.center  # (x, y) so c1[0] = x, c1[1] = y
    draw_rect_by_tuple(r1, col_black, stroke_size=stroke)

    r2 = Rect(width, height * 2, width, height)
    c2 = r2.center  # (x, y) so c2[0] = x, c2[1] = y
    draw_rect_by_tuple(r2, col_black, stroke_size=stroke)

    r3 = Rect(width * 2, height * 2, width, height)
    c3 = r3.center  # (x, y) so c3[0] = x, c3[1] = y
    draw_rect_by_tuple(r3, col_black, stroke_size=stroke)

    draw_button(c1[0], c1[1], "Red", col_black) #17: use variables from lines above to draw the button in the middle of the answer box
    draw_button(c2[0], c2[1], "Green", col_black)
    draw_button(c3[0], c3[1], "Blue", col_black)

    return r1, r2, r3


#17: define whether the eye is in the rectangle
def check_point_within_zones(zones, point):
    reset_saved_hit_values()

    # Check if collision/hit occurred
    for i, v in enumerate(zones): #17: v is the zone itself
        if v.collidepoint(point[0], point[1]): #17: collide is to see whether the eye is in the answer box
            if i == 0:
                draw_rect_alpha(SCREEN, col_gray_alpha, v)
                hit_detection('red')

            elif i == 1:
                draw_rect_alpha(SCREEN, col_gray_alpha, v)
                hit_detection('green')

            else:
                draw_rect_alpha(SCREEN, col_gray_alpha, v)
                hit_detection('blue')


def hit_detection(str_color): #for when you are looking at the boxes
    if hit[str_color] != 0.0:
        # Hit was previously set, check what the time difference was
        diff = time() - hit[str_color] #17: check the difference between the current time and the timer

        if diff > accepted_hit_timer: #17:if the difference is bigger than 1 sec, the answer is saved
            new_key_down_event = pg.event.Event(pg.locals.KEYDOWN, key=KEYS[str_color], mod=pg.locals.KMOD_NONE)
            pg.event.post(new_key_down_event)

        if DEBUG:
            print(f"DEBUG: {str_color} diff is: {diff}")
    else: #17: for if the timer has not reached 1 sec yet
        # If you look at red and then at blue, the timer for red is reset to 0 seconds
        for key in hit.keys():
            if key != str_color:
                hit[key] = 0.0
        # If hit is NOT set, set it to current time
        hit[str_color] = time()
        if DEBUG:
            print(f"DEBUG: {str_color} hit time set: {hit['blue']}")


def reset_saved_hit_values(int_multiply=4, reset_no_matter_what=False): #17: when the eye is 4 times in the box without being there long enough for the program to see it as an answer, the 'answering timer' is reset to 0 (below 4 times the eye-answering is summed in the timer)
    # Loop through the key (red, green, blue) value pairs
    for key, value in hit.items():
        # If a time value has been set, and this was more than x seconds ago
        if value != 0.0:
            if reset_no_matter_what:
                hit[key] = 0.0
            else:
                if time() - hit[key] > int_multiply * accepted_hit_timer:
                    hit[key] = 0.0


def check_answer_within_acceptable_timeframe(time_when_presented):
    if time() - time_when_presented > fail_timer:
        return True
    return False


def draw_button(xpos, ypos, label, color):
    text = font_small.render(label, True, color, BACKGR_COL)
    text_rectangle = text.get_rect()
    text_rectangle.center = (xpos, ypos)
    SCREEN.blit(text, text_rectangle)


def draw_stroop_welcome():
    text_surface = font.render("STROOP Experiment", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 150)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Press Spacebar to continue", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 300)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("If you want to restart the test at any moment use the Backspace key", True,
                                     col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 350)
    SCREEN.blit(text_surface, text_rectangle)


def draw_stimulus(color, word):
    text_surface = font.render(word, True, COLORS[color], col_gray)
    text_rectangle = text_surface.get_rect()
    # text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 100)
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, SCREEN_SIZE[1] / 3.0)
    SCREEN.blit(text_surface, text_rectangle)


def draw_failed():
    text_surface = font_small.render(f"Failed to respond within {fail_timer} seconds!", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 150)
    SCREEN.blit(text_surface, text_rectangle)

    text_surface = font_small.render("Press Spacebar to continue", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 300)
    SCREEN.blit(text_surface, text_rectangle)


def draw_feedback(correct, reaction_time):
    if correct:
        text_surface = font_small.render("correct", True, col_black, BACKGR_COL)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 150)
        SCREEN.blit(text_surface, text_rectangle)
        text_surface = font_small.render(str(int(reaction_time * 1000)) + "ms", True, col_black, BACKGR_COL)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 200)
        SCREEN.blit(text_surface, text_rectangle)
    else:
        text_surface = font_small.render("Wrong key!", True, col_red, BACKGR_COL)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 150)
        SCREEN.blit(text_surface, text_rectangle)

    text_surface = font_small.render("Press Spacebar to continue", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 300)
    SCREEN.blit(text_surface, text_rectangle)


def draw_goodbye():
    text_surface = font_small.render("END OF THE EXPERIMENT", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 150)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Close the application.", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 200)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Press 's' to save results to file", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 250)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Press backspace to restart stroop test", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0] / 2.0, 300)
    SCREEN.blit(text_surface, text_rectangle)


main()
