## YETA 1 Slideshow with YET
## Input: Images and Stimuli.csv file
## Results = table with Part, Stimulus, time, eye tracking coordinates

import sys
import os
import time
import logging as log
import pygame as pg
import pandas as pd
from pygame.locals import *

"""
Name of file containing the Stimuli. By switching the stimulus file,
different versions of the experiment can be tested. A shortened stimulus 
file speeds up the testing cycles.
"""

import libyeti14 as yeti14 ## importing everything from the module
from libyeti14 import draw_text ## we will use this a lot
"""
The Yeti14 engine provides classes for measuring and recording ET data,
handling a set of stimuli and calibrating the eye tracker.
"""


def main():

    # Connecting YET

    Yet = yeti14.YETI(USB, SURF)
    if(not Yet.connected):
        log.error("YET could not connect with USB " + str(USB))
        sys.exit()
    else:
        log.info("YET connected with " + str(Yet.fps) + " FPS and resolution " + str(Yet.frame_size))
    """
    Here a Yet object is created by connecting to the USB device. Usually, Yet is number 1,
    whereas teh webcam is 0.
    If this fails, try another USB device (0,1,2,3, ...) by setting the variable above.
    """        

    # Reading picture infos
    if os.path.isfile(STIM_PATH):
        STIMS = yeti14.StimulusSet(STIM_PATH)
        log.info(str(STIMS.n()) + " stimuli loaded")
    else:
        log.error(STIM_PATH  + ' not found. CWD: ' + os.getcwd())
        sys.exit()
    """
    Information about stimuli is collected from the Stimuli-csv file and
    used to create a StimulusSet object (which is a list of Stimuli objects).
    If this fails, your Stimuli.csv file is not in place (folder Stimuli), 
    or you are running this program in interactive mode (like working with R).
    """

    ## Calibration screens
    Cal = yeti14.Calib(SURF)
    log.info("Calibration screen loaded with " + str(Cal.n()) + " targets")
    """
    A calibration object is created with a default 3x3 grid
    """

    QCal = yeti14.Calib(SURF, pro_positions=[0.5, 0.5])
    """
    The quick calibration is created with one center target. Target positions are given
    as proportions of the screen surface.
    """


    ## Initial State
    try:
        Yet.init_eye_detection(EYECASC)
        log.info("Eye detection initialized: " + str(Yet.connected))
    except:
        log.error("Eye detection could not be initialized with Haar cascade " + EYECASC)
        sys.exit()
    
    STATE = "Detect"
    """
    The initial state is to detect eyes in the Yet stream using a Haar cascade. 
    This comes as a file with the Yeta_1 package. If this file is not found, 
    initialization fails and an error message is written to the log file.

    When the initialization succeeds, Yet can from now on use the detect_eye() method.
    This is not necessary to build an eye tracker, but a nice feature.
    """
    
    ## FAST LOOP
    while True:
    
        for event in pg.event.get():
            key_forward = event.type == KEYDOWN and event.key == K_SPACE
            key_back = event.type == KEYDOWN and event.key == K_BACKSPACE
            """
            The event handler loop is working off the queue of events 
            that have arrived since the last visit. As Yeta_1 only uses two keys, 
            we do a pre-classification into back and forward. This makes the following 
            code easier to read. Also, if you wanted to change the back and forward keys, 
            it can be changed in this one place.
            
            The following conditional creates state transitions (it really is only one).
            As we will see down below, frame processing and graphical display solely 
            depend on the state.
            """
            
            # Interactive transitionals (IT)

            if STATE == "Detect":
                if Yet.eye_detected:
                    if key_forward:
                        Yet.reset()
                        Cal.reset()
                        STATE = "Calibration"
                """
                The eye_detected state is set in the frame processing section, below.
                When in the present frame an eye is detected and the user presses Forward
                the program moves on to the calibration routine.
                """

            elif STATE == "Calibration":
                if key_forward:
                    Yet.update_frame()
                    Yet.update_eye_frame()
                    Yet.update_quad_bright()
                    Yet.record_calib_data(Cal.active_pos())
                    """
                    During the calibration the frame processing is shut off, which can be seen in the frame processing
                    section. On key press, one frame is captured, the quad-bright meaures are taken and added 
                    to the training set.

                    After that the program checks whether their are remaining calibration points. 
                    If so, the calibration advances.
                    If the sequence is complete, the recorded calibration data is used for training the Yeti. 
                    From this point on, Yet can produce eye coordinates relative to the screen.
                    """
                    if Cal.remaining() > 0:
                        Cal.next()
                        STATE = "Calibration"
                    else:
                        Yet.train()
                        STATE = "Validate"
                    log.info(STATE)
                elif key_back:
                    STATE = "Detect"
            elif STATE == "Validate":
                """
                After validation the program moves on prepareStimulus, which is an invisible state. 
                An automatic transitional down below takes care of this step
                """
                if key_forward:
                    STATE = "prepareStimulus"
                elif key_back:
                    Cal.reset()
                    Yet.reset()
                    STATE = "Calibration"
            elif STATE == "Quick":
                """
                The Quick calibration state succeeds the invisble state prepareStimulus. One reason for that is
                that the quick calibration uses a blurred version of the stimulus (see Presentitionals), 
                so it has to be available at this point.
                """
                if key_forward:
                    Yet.update_offsets(QCal.active_pos())
                    t_stim_started = time.time()
                    STATE = "Stimulus"
                    """
                    When the quick calibration is complete, Yet updates its offsets
                    and moves on to the stimulus presentation. When the user presses the key, 
                    the time is started. That is why we want the stimulus already loaded at this point.
                    """
            elif STATE == "Thank You":
                if key_forward:
                    Yet.release()
                    pg.quit()
                    sys.exit()
            if event.type == QUIT:
                """
                This conditional makes sure the program quits gracefully at any time,
                when the user presses Escape or closes the window.
                """

                Yet.release()
                pg.quit()
                sys.exit()
                

        # Automatic transitionals (AT)
        """
        Automatic transitionals are used to react to internal states of the system.
        
        Here are two different examples of using ATs, completing a complex task and reacting to time.
        """

        if STATE == "prepareStimulus":
            ret, Stim = STIMS.next()
            Stim.load(SURF)
            if ret:
                STATE = "Quick"
            else:
                log.error("Could not load next Stimulus")
                sys.exit()
            """
            This ATC completes a complex computation and immediatly moves on to the next state.
            
            Before a stimulus can be shown on the screen it has to be loaded from the hard drive 
            and pre-processed. All this takes time which we don't want to confound with the presentation time.
            """
            
        if STATE == "Stimulus":
            elapsed_time = time.time() - t_stim_started # elapsed time since STATE == "Stimulus" started
            if elapsed_time > SLIDE_TIME: #presented longer then defined: trial is over
                Yet.data.to_csv(RESULT_FILE, index = False) ## auto save results after every slide
                if STIMS.remaining() > 0:  # if images are left, got to quick cal
                    Yet.reset_offsets()
                    STATE = "prepareStimulus"
                    log.info(STATE)
                else:
                    STATE = "Thank You"
                    log.info(STATE)
            
            """
            This ATC controls the presentation time. 
            When the time is passed, collected data is saved to a CSV file. 
            When there are more stimuli, the program moves on to prepare
            another stimulus.
            """
    

        # FRAME PROCESSING
        
        """
        Frame processing basically is a data processing stack, also very much like
        a pipeline in R. However, in this case we first have to build the stack, 
        and that is why the Yeti class exposes all individual processing steps.
        To make sure the processing steps are only computed once, all steps come
        as updatze functions. These set the respective attribute, 
        which can later be retrieved multiple times without being calculated again.
        This saves processing time, but the developer has to know the steps
        and apply them correctly.
        """

        if STATE == "Detect":
            Yet.update_frame()
            Yet.detect_eye()
            if Yet.eye_detected:
                Yet.update_eye_frame()
            """
            During the eye detection state, the Yet frame is continiously updated 
            and undergoes eye detection. This is a very computing intensive task.
            This is why we don't use it throughout the whole experiment. 
            (And because it makes the signal more shakey).
            """
        elif STATE == "Validate" or STATE == "Quick":
            Yet.update_frame()
            Yet.update_eye_frame()
            Yet.update_quad_bright()
            Yet.update_eye_pos()
            """
            The Validate state runs through the whole prediction statck. A trained model is required.
            As a result, the draw() method becomes available, which is used in the respective Presentitional.
            """
        elif STATE == "Stimulus":
            """
            In the Stimulus state, the prediction stack is further extended to produce 
            coordinates relative to the original stimulus dimensions (in pixel), 
            as the stimulus may have been centered and scaled. Then the data is internally recorded by the
            Yet object.
            """
            Yet.update_frame()
            Yet.update_eye_frame()
            Yet.update_quad_bright()
            Yet.update_eye_pos()
            Yet.update_eye_stim(Stim)
            Yet.record(EXP_ID + EXPERIMENTER, PART_ID, Stim.file)

        # Presentitionals
        SURF.fill(BACKGR_COL)
        """
        Remember, we are in the fast while loop. 
        With every round the display is refreshed by painting it over with the 
        background color.
        """

        if STATE == "Detect":
            if Yet.eye_detected:
                Img = yeti14.frame_to_surf(Yet.eye_frame, (int(SURF_SIZE[0] * .5), int(SURF_SIZE[1] * .5)))
                draw_text("Eye detected!", SURF, (.1, .85), FONT, color = col_green)
                draw_text("Space to continue", SURF, (.1, .9), Font)
            else:
                Img = yeti14.frame_to_surf(Yet.frame, (int(SURF_SIZE[0] * .5), int(SURF_SIZE[1] * .5)))
                draw_text("Trying to detect an eye.", SURF, (.1, .85), FONT)
            SURF.blit(Img, (int(SURF_SIZE[0] * .25), int(SURF_SIZE[1] * .25)))
            """
            The Detect srceen is dynamic in that it changes when an eye detected,
            which can change from one Yet frame to the next.
            """
        elif STATE == "Calibration":
            Cal.draw()
            draw_text("Follow the orange circle and press Space.", SURF, (.1, .9), Font)
            """
            The Calib class brings its own draw function. 
            """
        elif STATE == "Validate":
            draw_text("Space: continue", SURF, (.1, .9), Font)
            draw_text("Backspace: redo the calibration.", SURF, (.1, .95), Font)
            Yet.draw_follow(SURF)
            """
            The Yet class brings a draw function that produces a following dot.
            """
        elif STATE == "Stimulus":
            Stim.draw()
            """
            The Stimulus class brings its own draw function, which takes care 
            of positioning and scaling of the original image.
            """
        elif STATE == "Quick":
            Stim.draw_preview()
            QCal.draw()
            Yet.draw_follow(SURF)
            draw_text("Look at the orange circle and press Space.", SURF, (.05, .75), Font)
            """
            During the Quick state  a blurred preview of the stimulus is shown in the backgroud.
            This way, the quick calibration can also regard the difference in screen brightness.
            Under certain lighting conditions, a the change in screen brightness, when the stimulus is presented,
            can severely bias the measures, as it influences the brightness measures.
            """

        elif STATE == "Thank You":
            draw_text("Thank you for taking part!", SURF, (.1, .5), FONT)
            draw_text("Press Space to end the program. Data has been saved", SURF, (.1, .8), Font)
            

        # update the screen to display the changes you made
        pg.display.update()


def read_config(path = "Config.csv"):
    global USB, EXP_ID, EXPERIMENTER
    global SURF_SIZE, SLIDE_TIME, STIM_FILE
    global CONFIG 

    CONFIG = dict()
    Tab = pd.read_csv(path)
    for index, row in Tab.iterrows():
        CONFIG[row[0]] = row[1]
    USB = int(CONFIG["USB"])
    EXP_ID = str(CONFIG["EXP_ID"])
    EXPERIMENTER = str(CONFIG["EXPERIMENTER"])
    SURF_SIZE = (int(CONFIG["WIDTH"]), int(CONFIG["HEIGHT"]))
    SLIDE_TIME = float(CONFIG["SLIDE_TIME"])
    STIM_FILE = CONFIG["STIM_FILE"]



def setup():
    """
    Creates global variables for Yeta_1 and changes the working directory
    """
    global EXP_ID, EXPERIMENTER
    global YETA, YETA_NAME
    global WD, STIM_DIR, STIM_PATH, RESULT_DIR,\
         PART_ID, RESULT_FILE, EYECASC
    
    ## Meta data of Yeta
    YETA = 1
    YETA_NAME = "Yeta" + str(YETA)

    ## Paths and files
    WD = os.path.dirname(sys.argv[0])
    os.chdir(WD)
    """Working directory set to location of yeta_1.py"""

    STIM_DIR = os.path.join(WD, "Stimuli")
    """Directory where stimuli reside"""
    STIM_PATH = os.path.join(STIM_DIR, "Stimuli.csv")
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


def init_pygame():
    # Pygame init
    pg.init()
    global FONT, Font, font
    global col_black, col_green, col_white, BACKGR_COL
    global SURF
    
    FONT = pg.font.Font('freesansbold.ttf', int(min(SURF_SIZE) / 20))
    Font = pg.font.Font('freesansbold.ttf', int(min(SURF_SIZE) / 40))
    font = pg.font.Font('freesansbold.ttf', int(min(SURF_SIZE) / 60))
    pg.display.set_mode(SURF_SIZE)
    pg.display.set_caption(YETA_NAME)
    SURF = pg.display.get_surface()

    # Colour definitions
    col_black = (0, 0, 0)
    col_green = (0, 255, 0)
    col_white = (255, 255, 255)

    BACKGR_COL = col_white

read_config()
setup()
init_pygame()
main()
