## YETA 1 Slideshow with YET
## Input: Images and Stimuli.csv file
## Results = table with Part, Stimulus, time, eye tracking coordinates

## Basic configuration
USB = 1
"""USB camera device number for YET (typically 1, sometimes 0 or 2)"""
EXP_ID = "UV22"
"""Identifier for experiment"""
EXPERIMENTER = "MS"
"""Experimenter ID"""
SURF_SIZE = (1000, 1000)
"""Screen dimensions"""
SLIDE_TIME = 4
"""Presentation time per stimulus"""

import sys
import os
import time
import logging as log
import pygame as pg
from pygame.locals import *
# own classes
import libyeti14 as yeti14
import libyeta1 as yeta1

def main():

    """GLOBAL VARIABLES"""
    
    # Connecting YET
    Yet = yeti14.YET(USB, SURF)
    if(not Yet.connected):
        log.error("YET could not connect with USB " + str(USB))
        sys.exit()
    else:
        log.info("YET connected with " + str(Yet.fps) + " FPS and resolution " + str(Yet.frame_size))

    # Reading picture infos
    if os.path.isfile(STIM_INFO):
        STIMS = yeti14.StimulusSet(STIM_INFO)
        log.info(str(STIMS.n()) + " stimuli loaded")
    else:
        log.error(STIM_INFO  + ' not found. CWD: ' + os.getcwd())
        sys.exit()

    ## Calibration screens
    Cal = yeti14.Calib(SURF)
    log.info("Calibration screen loaded with " + str(Cal.n()) + " targets")
    QCal = yeti14.Calib(SURF, pro_positions=[0.5])


    ## Initial State
    STATE = "Detect"
    
    try:
        Yet.init_eye_detection(EYECASC)
        log.info("Eye detection initialized: " + str(Yet.connected))
    except:
        log.error("Eye detection could not be initialized with Haar cascade " + EYECASC)

    ## FAST LOOP
    while True:

        ## EVENT HANDLING
        for event in pg.event.get():
            key_forward = event.type == KEYDOWN and event.key == K_SPACE
            key_back = event.type == KEYDOWN and event.key == K_BACKSPACE
            
            # Interactive transition conditionals (ITC)

            if STATE == "Detect":
                if Yet.eye_detected:
                    if key_forward:
                        Yet.reset()
                        Cal.reset()
                        STATE = "Target"
            elif STATE == "Target":
                if key_forward:
                    Yet.update_frame()
                    Yet.update_eye_frame()
                    Yet.update_quad_bright()
                    Yet.record_calib_data(Cal.active_pos())
                    if Cal.remaining() > 0:
                        Cal.next()
                        STATE = "Target"
                    else:
                        Yet.train()
                        STATE = "Validate"
                    log.info(STATE)
                elif key_back:
                    STATE = "Detect"
            elif STATE == "Validate":
                if key_forward:
                    STATE = "prepareStimulus"
                elif key_back:
                    Cal.reset()
                    Yet.reset()
                    STATE = "Target"
            elif STATE == "Quick":
                if key_forward:
                    Yet.update_offsets(QCal.active_pos())
                    STATE = "prepareStimulus"
            elif STATE == "Thank You":
                if key_forward:
                    Yet.release()
                    pg.quit()
                    sys.exit()
            if event.type == QUIT:
                Yet.release()
                pg.quit()
                sys.exit()



        # Automatic transitionals (ATC)
        if STATE == "prepareStimulus":
            ret, Stim = STIMS.next()
            Stim.load(SURF)
            if ret:
                STATE = "Stimulus"
                t_stim_started = time.time()
            else:
                log.error("Could not load next Stimulus")
                sys.exit()
            
        if STATE == "Stimulus":
            elapsed_time = time.time() - t_stim_started # elapsed time since STATE == "Stimulus" started
            if elapsed_time > SLIDE_TIME: #presented longer then defined: trial is over
                Yet.data.to_csv(RESULT_FILE, index = False) ## auto save results after every slide
                if STIMS.remaining() > 0:  # if images are left, got to quick cal
                    Yet.reset_offsets()
                    STATE = "Quick"
                    # STATE = "prepareStimulus"
                    log.info(STATE)    
                else:
                    STATE = "Thank You"
                    log.info(STATE)
    

        # FRAME PROCESSING


        if STATE == "Detect":
            Yet.update_frame()
            Yet.detect_eye()
            if Yet.eye_detected:
                Yet.update_eye_frame()
        
        if STATE == "Validate" or STATE == "Quick":
            Yet.update_frame()
            Yet.update_eye_frame()
            Yet.update_quad_bright()
            Yet.update_eye_pos()

        if STATE == "Stimulus":
            Yet.update_frame()
            Yet.update_eye_frame()
            Yet.update_quad_bright()
            Yet.update_eye_pos()
            Yet.update_stim_pos(Stim)
            Yet.record(EXP_ID + EXPERIMENTER, PART_ID, Stim.file)

        # Presentitionals
        SURF.fill(BACKGR_COL)

        if STATE == "Detect":
            if Yet.eye_detected:
                Img = yeti14.frame_to_surf(Yet.eye_frame, (int(SURF_SIZE[0] * .5), int(SURF_SIZE[1] * .5)))
                yeta1.draw_text("Eye detected!", SURF, (.1, .85), FONT, color = col_green)
                yeta1.draw_text("Space to continue", SURF, (.1, .9), Font)
            else:
                Img = yeti14.frame_to_surf(Yet.frame, (int(SURF_SIZE[0] * .5), int(SURF_SIZE[1] * .5)))
                yeta1.draw_text("Trying to detect an eye.", SURF, (.1, .85), FONT)
            SURF.blit(Img, (int(SURF_SIZE[0] * .25), int(SURF_SIZE[1] * .25)))
        elif STATE == "Target":
            Cal.draw()
            yeta1.draw_text("Follow the orange circle and press Space.", SURF, (.1, .9), Font)
        elif STATE == "Validate":
            yeta1.draw_text("Space: continue", SURF, (.1, .9), Font)
            yeta1.draw_text("Backspace: redo the calibration.", SURF, (.1, .95), Font)
            Yet.draw_follow(SURF)
        elif STATE == "Stimulus":
            Stim.draw()
            Yet.draw_follow(SURF)
        elif STATE == "Quick":
            QCal.draw()
            Yet.draw_follow(SURF)
            yeta1.draw_text("Look at the orange circle and press Space.", SURF, (.05, .75), Font)
        elif STATE == "Thank You":
            yeta1.draw_text("Thank you for taking part!", SURF, (.1, .5), FONT)
            yeta1.draw_text("Press Space to end the program. Data has been saved", SURF, (.1, .8), Font)

        # update the screen to display the changes you made
        pg.display.update()


def setup():
    """
    Creates global variables for Yeta_1 and changes the working directory
    """
    global EXP_ID, EXPERIMENTER
    global YETA, YETA_NAME
    global WD, STIM_DIR, STIM_INFO, RESULT_DIR,\
         PART_ID, RESULT_FILE, EYECASC
    global col_black, col_green, col_white, BACKGR_COL

    ## Meta data of Yeta
    YETA = 1
    YETA_NAME = "Yeta" + str(YETA)

    ## Paths and files
    WD = os.path.dirname(sys.argv[0])
    os.chdir(WD)
    """Working directory set to location of yeta_1.py"""

    STIM_DIR = os.path.join(WD, "Stimuli")
    """Directory where stimuli reside"""
    STIM_INFO = os.path.join(STIM_DIR, "Stimuli_short.csv")
    """CSV file describing stimuli"""
    RESULT_DIR = "Data"
    """Directory where results are written"""
    PART_ID = str(int(time.time()))
    """Unique participant identifier by using timestamps"""
    RESULT_FILE = os.path.join(RESULT_DIR, YETA_NAME + "_" +\
         EXP_ID + EXPERIMENTER + PART_ID + ".csv")
    """File name for data"""
    EYECASC = "haarcascade_eye.xml"

    ##### Logging #####
    log.basicConfig(filename='Yet.log', level=log.INFO)

    # Colour definitions
    col_black = (0, 0, 0)
    col_green = (0, 255, 0)
    col_white = (255, 255, 255)

    BACKGR_COL = col_white


def init_pygame():
    # Pygame init
    pg.init()
    global FONT 
    global Font
    global font
    global SURF
    
    FONT = pg.font.Font('freesansbold.ttf', int(min(SURF_SIZE) / 20))
    Font = pg.font.Font('freesansbold.ttf', int(min(SURF_SIZE) / 40))
    font = pg.font.Font('freesansbold.ttf', int(min(SURF_SIZE) / 60))
    pg.display.set_mode(SURF_SIZE)
    pg.display.set_caption(YETA_NAME)
    SURF = pg.display.get_surface()

setup()
init_pygame()
main()
