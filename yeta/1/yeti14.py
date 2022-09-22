import os
import numpy as np
"""Numpy for data manipulation"""
from sklearn import linear_model as lm
"""Using linear models from Sklearn"""
import pandas as pd
"""Using Pandas data frames"""
# import csv
# """Reading and writing CSV files"""
from time import time
from pygame.draw import circle 

# CV
import cv2 as cv
"""OpenCV computer vision library"""

class Calib:
    """yeti14 calibration routines"""
    screen_size = ()
    rel_positions = ()
    targets = np.zeros(shape = (2, 0))
    active_target = int() 
    color = (160, 160, 160)
    active_color = (255, 120, 0)
    radius = 20
    stroke = 10

    def __init__(self, screen, rel_positions = [0.125, 0.5, 0.875]):
        self.screen = screen
        self.rel_positions = np.array(rel_positions)
        points = np.multiply([[self.screen_size[1]], [self.screen_size[0]]], 
                             rel_positions) # x and y positions
        points = np.round(points) # round the value to the nearest integer value
        points = points.astype(int) # make it an integer
        self.targets = np.empty([1, 2]) # create an array of correct dimensions for the targets
        for i in range(len(points[1])):
            values_y = points[0, i] * np.ones([len(points[0]), 1]) # Make a y value array
            values_x = np.transpose(points[[1], :]) # make a x value array
            values_x = np.append(values_x, values_y, axis=1) # combine arrays
            self.targets = np.append(self.targets, values_x, axis=0) # put into the targets array
            self.targets = np.delete(self.targets, 0, axis=0)
    
    def active_pos(self):
        return self.targets[:, self.active_target]

    def len(self):
        return len(self.rel_positions)

    def screen_size(self):
        return self.screen.get_window_size()
    
    def draw_target(self, index, active = False):
        pos = self.targets[:,index]
        if active:
            color = self.active_color
        else:
            color = self.color
        circle(self.screen, color, pos, self.radius, self.stroke)

    def draw_calib_screen(self, active_target = None):
        cnt = 0
        for target in self.targets:
            pos = list(map(int, target))
            self.draw_target(pos, active = cnt == active_target)
            cnt += 1
            circle(self.screen, self.color, pos, self.radius, self.stroke)

    def draw_quick_screen(self):
        w, h = self.screen_size()
        pos = (int(w/2), int(h/2))
        self.draw_target(active = True)


class YET:
    """Using the YET device"""
    width = 0
    height = 0
    fps = 0
    dim = (0,0)
    frame = []
    new_frame = False
    connected = False
    cascade = False
    eye_detection = False
    eye_detected = False
    eye_frame_coords = (0,0,0,0) # make array
    eye_frame = False
    calib_data = np.zeros(shape=(0, 6))
    quad_bright = (0,0,0,0) # make array
    eye_pos_raw = (0, 0)
    offsets = (0,0) # make array
    scale_image = (1,1)
    eye_pos = (0, 0)
    data = pd.DataFrame(columns = ("Experiment","Part", "Stimulus", "time", "x", "y", "x_offset", "y_offset") , 
                       dtype = "float64")

    def __init__(self, usb):
        """
        YET constructor

        :param usb USB port, usually 1
        :type usb int
        """
        self.data["Experiment"].astype("category")
        self.data["Part"].astype("category")
        self.data["Stimulus"].astype("category")

        self.device = cv.VideoCapture(usb)
        self.connected = True
            
        if self.connected:
            self.usb = usb
            self.width = int(self.device.get(cv.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.device.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.device.get(cv.CAP_PROP_FPS)
            self.dim = (self.width, self.height)
            
    def release(self):
        self.device.release()
    
    def update_frame(self):
        """
        Update the eye frame based on eye detection
        """
        self.new_frame, self.frame = self.device.read()
        return(self.new_frame)

    def init_eye_detection(self, cascade_file):
        """
        Initialize eye detection

        :param model Haar cascade file
        """
        self.eye_detection = False
        self.cascade = cv.CascadeClassifier(cascade_file)
        self.eye_detection = True

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

    def save_data(self):
        """
        Saves recorded eye tracking data

        """

        
