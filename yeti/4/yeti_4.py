## Minimal stream viewer using PyGame/OpenCV

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
import pygame
from pygame.locals import *
from pygame.compat import unichr_, unicode_

##### VARIABLES #####

# CV

log.basicConfig(filename='webcam.log',level=log.INFO)

YET = cv2.VideoCapture(1)

if not YET.isOpened():
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

pygame.init()
pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Draw images")

screen = pygame.display.get_surface()
screen.fill(BACKGR_COL)

font = pygame.font.Font(None, 60)

def main():
    
    ## PG
    
    print("Canvas size is (" + str(SCREEN_SIZE[0]) + "," + str(SCREEN_SIZE[1]) + ")")
    print("(0,0) is the upper left corner")    
    while True:
        # CV
        
        ret, Frame = YET.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        F_rgb = cv2.cvtColor(Frame,cv2.COLOR_BGR2RGB) # convert BGR (CV2) to RGB (Pygame)
        F_img = np.rot90(F_rgb) # rotate coordinate system
        
        # PG
        pygame.display.get_surface().fill(BACKGR_COL) 
        for event in pygame.event.get():
            # Interactive transition conditionals (ITC)
            # always include transition for quit events
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        Img = pygame.surfarray.make_surface(F_img)
        Img = pygame.transform.smoothscale(Img, (900, 700))
        screen.blit(Img,(50,50))
        # update the screen to display the changes you made
        pygame.display.update()


main()
YET.release()
cv2.destroyAllWindows()
