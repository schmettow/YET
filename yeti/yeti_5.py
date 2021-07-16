## CV2/Pygame Streamplayer with snaphot function

import sys
import logging as log
import datetime as dt
from time import sleep
import random
# DS
import numpy as np
# CV
import cv2
# PG
import pygame as pg
from pygame.locals import *
from pygame.compat import unichr_, unicode_

##### Preparations #####

# CV

log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(1)

if not video_capture.isOpened():
        print('Unable to load camera.')
        exit()

# PG

col_black = (0, 0, 0)
col_gray = (120, 120, 120)
col_red = (250, 0, 0)
col_green = (0, 255, 0)
col_yellow = (250, 250, 0)

col_red_dim = (120, 0, 0)
col_green_dim = (0, 60, 0)
col_yellow_dim = (120,120,0)

BACKGR_COL = col_black
SCREEN_SIZE = (1000, 800)

pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption("Draw images")

screen = pg.display.get_surface()
screen.fill(BACKGR_COL)

font = pg.font.Font(None, 60)

## FAST LOOP

def main():
    
    STATE = "Stream"
    ## PG
    
    print("Canvas size is (" + str(SCREEN_SIZE[0]) + "," + str(SCREEN_SIZE[1]) + ")")
    print("(0,0) is the upper left corner")    
    while True:
        # PG
        pg.display.get_surface().fill(BACKGR_COL) 
        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Stream":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Captured"
                    print("Captured")
            elif STATE == "Captured":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Stream"
                    print("Stream")
            if event.type == QUIT:
                video_capture.release()
                pg.quit()
                sys.exit()
            
        # Conditional Processing
        if STATE == "Stream":
            ret, Frame = video_capture.read(1)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        if STATE == "Captured":
            captured_Frame = Frame

        # Presentitionals
        if STATE == "Stream":
            Img = frame_to_surf(Frame, (900, 700))
            screen.blit(Img,(50,50))

        if STATE == "Captured":
            Img = frame_to_surf(captured_Frame, (900, 700))
            screen.blit(Img,(50,50))
        
        # update the screen to display the changes you made
        pg.display.update()


## Converts a CV2 framebuffer into Pygame image (surface!)
def frame_to_surf(frame, dim):
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # convert BGR (CV2) to RGB (Pygame)
    img = np.rot90(img) # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf

main()
