import os
import numpy as np
"""Numpy for data manipulation"""
import itertools
from sklearn import linear_model as lm
"""Using linear models from Sklearn"""
import pandas as pd
"""Using Pandas data frames"""
# import csv
# """Reading and writing CSV files"""
from time import time, sleep
import pygame as pg
from pygame.draw import circle
import logging as log

# CV
import cv2 as cv
"""OpenCV computer vision library"""

def main():
    print("main")
    pg.init()
    pg.display.set_mode((800, 800))
    SCREEN = pg.display.get_surface()
    Cal = Calib(SCREEN)
    print(str(Cal.targets))
    print("active: " + str(Cal.active))
    Cal.draw()
    pg.display.update()
    sleep(2)

class Calib:
    """yeti14 calibration"""
    color = (160, 160, 160)
    active_color = (255, 120, 0)
    radius = 20
    stroke = 10


    """
    Creates a square calib surface using relative positions
    ...

    Attributes
    ----------
    surface : pygame.Surface
        a Pygame surface object for drawing
    rel_positions : tuple[int]
        relative target positions used for creating a square calibration surface
    targets : numpy.array
        actual target positions (x,y)
    active : int
        index of the active targets

    Methods
    -------
    active_pos()
        returns the coordinates of the active targets
    reset()
        resets the active position to 0
    n()
        returns the number of targets to
    remaining()
        returns the number of remaining targets
    next()
        advances active position by 1
    draw()
        draws the calibration surface
    
    """

    def __init__(self, surface, rel_positions = (0.125, 0.5, 0.875)):
        self.surface = surface
        self.surface_size = np.array(self.surface.get_size())
        self.rel_positions = np.array(rel_positions)
        x_pos = self.rel_positions * self.surface_size[0]
        y_pos = self.rel_positions * self.surface_size[1]
        self.targets = np.array(list(itertools.product(x_pos,y_pos)))
        self.active = 0
    
    def active_pos(self):
        return self.targets[self.active]

    def reset(self):
        self.active = 0

    def n(self):
        return len(self.targets[:,0])

    def remaining(self):
        return self.n() - self.active - 1 

    def next(self):
        if self.remaining():
            this_target = self.targets[self.active]
            self.active += 1
            return True, this_target
        else:
            return False, None

    def draw(self):
        index = 0
        for target in self.targets:
            pos = list(map(int, target))
            if index == self.active:
                color = self.active_color
            else:
                color = self.color
            index += 1
            circle(self.surface, color, pos, self.radius, self.stroke)




class YET:
    """
    Class for using a USB camera as an eye tracking device
    ...

    Attributes
    ----------
    usb : int
        number of the USB camera to connect
    device : cv2.VideoCapture
        camera device after connection
    connected : bool
        if camera device is connected
    fps : int
        Frame refresh rate of camera
    frame_size : tuple(int)
        width and height of frames delivered by connected camera
    calib_data : pandas.DataFrame
        Data frame collecting calibration data
    data : pandas.DataFrame
        Data frame to collect recorded eye positions
    
    surface : pygame.Surface
        a Pygame surface object for drawing
    rel_positions : tuple[int]
        relative target positions used for creating a square calibration surface
    targets : numpy.array
        actual target positions (x,y)
    active : int
        index of the active targets

    Methods
    -------
    release()
        returns the coordinates of the active targets
    init_eye_detection()
        resets the active position to 0
    update_frame()
        returns the number of targets to
    detect_eye()
        returns the number of remaining targets
    update_eye_frame()
        advances active position by 1
    update_quad_bright()
        draws the calibration surface
    record_calib_data()
        adds current measures to calibration data
    train()
        trains the eye tracker using recorded calibration data
    reset()
        resets calibration data, offsets and data
    """
    
    frame = []
    new_frame = False
    connected = False
    cascade = False
    eye_detection = False
    eye_detected = False
    eye_frame_coords = (0,0,0,0) # make array
    eye_frame = False
    quad_bright = (0,0,0,0) # make array
    eye_pos_raw = (0, 0)
    offsets = (0,0) # make array
    scale_image = (1,1)
    eye_pos = (0, 0)

    def __init__(self, usb):
        """
        YET constructor

        :param usb USB port, usually 1
        :type usb int
        """
        self.connected = False
        self.usb = usb
        try:
            self.device = cv.VideoCapture(self.usb)
            self.connected = True
        except:
            log.error("Could not connect USB device " + self.usb)
            
        if self.connected:
            self.fps = self.device.get(cv.CAP_PROP_FPS)
            self.frame_size = (int(self.device.get(cv.CAP_PROP_FRAME_WIDTH)),
                               int(self.device.get(cv.CAP_PROP_FRAME_HEIGHT)))
            self.calib_data = np.zeros(shape=(0, 6))
            self.data = pd.DataFrame(columns = ("Experiment","Part", "Stimulus", "time", "x", "y", "x_offset", "y_offset") , 
                       dtype = "float64")
            self.data["Experiment"].astype("category")
            self.data["Part"].astype("category")
            self.data["Stimulus"].astype("category")

    def release(self):
        self.device.release()

    def init_eye_detection(self, cascade_file):
        """
        Initialize eye detection

        :param model Haar cascade file
        """
        self.eye_detection = False
        self.cascade = cv.CascadeClassifier(cascade_file)
        self.eye_detection = True

    def update_frame(self):
        """
        Update the eye frame based on eye detection
        """
        self.new_frame, self.frame = self.device.read()
        return(self.new_frame)

    def detect_eye(self):
        """
        Updates the position and size of the eye frame
        """
        if self.new_frame:
            Eyes = self.cascade.detectMultiScale(
                self.frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)) ## <-- parametrize me
            if len(Eyes) > 0:
                self.eye_detected = True
                self.eye_frame_coords = Eyes[0]
            else:
                self.eye_detected = False
        return(self.eye_detected, self.eye_frame_coords)

    
    def update_eye_frame(self):
        """
        Trims the frame obtained by eye detection
        """
        if self.new_frame:
            x, y, w, h = self.eye_frame_coords
            self.eye_frame = self.frame[y:y + h, x:x + w] 
            self.eye_frame = cv.cvtColor(self.eye_frame, cv.COLOR_BGR2GRAY)
        return(self.eye_frame)
    
    def update_quad_bright(self):
        """
        Updating the quadrant brightness vector
        """
        if self.new_frame:
            w, h = np.shape(self.eye_frame)
            b_NW = np.mean(self.eye_frame[0:int(h / 2), 0:int(w / 2)])
            b_NE = np.mean(self.eye_frame[int(h / 2):h, 0:int(w / 2)])
            b_SW = np.mean(self.eye_frame[0:int(h / 2), int(w / 2):w])
            b_SE = np.mean(self.eye_frame[int(h / 2):h, int(w / 2):w])
            self.quad_bright = (b_NW, b_NE, b_SW, b_SE)
        return(self.quad_bright)
    
    def record_calib_data(self, target_pos):
        """
        Record the present quad brightness for training the model

        :param target_pos (x, y) position of calibration target
        """
        new_data = np.append(self.quad_bright, np.array(target_pos))
        self.calib_data = np.append(self.calib_data, [new_data], axis = 0)
        return(new_data)

    def train(self):
        """
        Trains the eye tracker

        :param data is a 6 (4 + 2) column array
        """
        Quad = self.calib_data[:, 0:4]
        Pos = self.calib_data[:, 4:6]
        model = lm.LinearRegression()
        model.fit(Quad, Pos)
        self.model = model
        return(self.model)

    def update_offsets(self, target_pos):
        """
        Updates the offset values based on current eye_pos and a given target position

        param target_pos = position of visual target
        """
        self.offsets = [target_pos[0] - self.eye_pos_raw[0], 
                        target_pos[1] - self.eye_pos_raw[1]]
        return(self.offsets)

    
    
    def update_eye_pos(self):
        """
        Predicts the eye coordinates

        """        
        quad = np.array(self.quad_bright)
        quad.shape = (1, 4)
        x, y = self.model.predict(quad)[0, :]
        self.eye_pos_raw = (x, y)
        self.eye_pos = (x + self.offsets[0], y + self.offsets[1])
        return([self.eye_pos_raw, self.eye_pos])

    def record(self, Exp_ID, Part_ID, Stim_ID):
        """
        Records the eye coordinates

        """
        new_data = pd.DataFrame({"Experiment": Exp_ID, 
                                    "Part": Part_ID,
                                    "Stimulus": Stim_ID, 
                                    "time" : time(),
                                    "x": self.eye_pos[0]/self.scale_image[0], 
                                    "y": self.eye_pos[1]/self.scale_image[1],
                                    "x_offset": self.offsets[0], 
                                    "y_offset": self.offsets[1]}, 
                                  index = [0])
        self.data = pd.concat([self.data, new_data])
        return(new_data)

    def reset_calib(self):
        self.calib_data = np.zeros(shape=(0, 6))
        del self.model

    def reset_offsets(self):
        self.offsets = (0,0)

    def reset_data(self):
        self.data = pd.DataFrame(columns = ("Experiment","Part", "Stimulus", "time", "x", "y", "x_offset", "y_offset") , 
                       dtype = "float64")

    def reset(self):
        self.reset_calib()
        self.reset_data()

    def draw_follow(self, surface):
        
        """
        Draws a circle to the current eye position

        Note that eye positions must be updated using the update methods
        """
        
        circle(surface, (120, 120, 0),  self.eye_pos, 12, 3)

        
        
def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img)  # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf

