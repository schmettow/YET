import os, random
import numpy as np
from numpy import array as ary
"""Numpy for data manipulation"""

import itertools
from sklearn import linear_model as lm

"""Using linear models from Sklearn"""
import pandas as pd

"""Using Pandas data frames"""
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

def draw_text(text: str, Surf: pg.Surface, rel_pos: tuple, Font: pg.font.Font, 
              color=(0, 0, 0), center=False):
    surf_size = Surf.get_size()
    x, y = np.array(rel_pos) * np.array(surf_size)
    rendered_text = Font.render(text, True, color)
    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()
    # this sets the x and why coordinates
    if center:
        box.center = (x, y)
    else:
        box.topleft = (x, y)
    # This puts the pre-rendered object on the surface
    Surf.blit(rendered_text, box)


class Stimulus:
  stim_dir = "Stimuli/"

  def __init__(self, entry):
    if isinstance(entry, pd.DataFrame):
        entry = entry.to_dict()
    self.file = entry["File"]
    self.path = os.path.join(self.stim_dir,  self.file)
    self.size = ary((entry["width"], entry["height"]))
    
  def load(self, surface: pg.Surface, scale = True):
    image = pg.image.load(self.path)
    # image = pg.image.convert()
    self.surface = surface
    self.surf_size = ary(self.surface.get_size())
    if scale:
        self.scale = min(self.surf_size / self.size)
        scale_to = ary(self.size * self.scale).astype(int)
        self.image = pg.transform.smoothscale(image, scale_to)    
        self.size = self.image.get_size()
    else:
        self.scale = 1
    self.pos = ary((self.surf_size - self.size)/2).astype(int)

  def draw(self):
    self.surface.blit(self.image, self.pos)

  
  def draw_preview(self):
    blur = ary(self.surf_size/4).astype('int') # 10% blur
    img = pg.surfarray.array3d(self.image)
    img = cv.blur(img, blur).astype('uint8')
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = pg.surfarray.make_surface(img)
    self.surface.blit(img, self.pos)
    
    
  
  def average_brightness(self):
    return pg.surfarray.array3d(self.image).mean()


class StimulusSet:
  def __init__(self, path):
    self.table = pd.read_csv(path)
    self.Stimuli = []
    for index, row in self.table.iterrows():
        this_stim = Stimulus(row)
        self.Stimuli.append(this_stim)
    self.active = 0

  def n(self):
    return len(self.Stimuli)

  def remaining(self):
    return len(self.Stimuli) - self.active

  def next(self):
    if self.active < len(self.Stimuli):
      this_stim = self.Stimuli[self.active]
      self.active += 1
      return True, this_stim
    else:
      return False, None

  def reset(self):
    self.active = 0
  
  def pop(self):
    return self.Stimuli.pop()

  def shuffle(self, reset = True):
    self.reset()
    random.shuffle(self.Stimuli) ## very procedural, brrr

        
def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img)  # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf







class YETI:
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
    pro_positions : tuple[int]
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
    
    frame = None
    new_frame = False
    connected = False
    cascade = False
    eye_detection = False
    eye_detected = False
    eye_frame_coords = (0,0,0,0) # make array
    eye_frame = []
    quad_bright = (0,0,0,0) # make array
    offsets = (0,0) # make array
    data_cols = ("Exp","Part", "Stim", "time", "x", "y", "x_pro", "y_pro") 

    def __init__(self, usb: int, surface: pg.Surface) -> None:
        """
        YETI constructor

        :param usb USB port, usually 1
        :type usb int
        """
        self.connected = False
        self.usb = usb
        self.surface = surface
        self.surf_size = self.surface.get_size()
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
            self.data = pd.DataFrame(columns = YETI.data_cols,
                       dtype = "float64")
            self.data["Exp"].astype("category")
            self.data["Part"].astype("category")
            self.data["Stim"].astype("category")
            self.update_frame()

    def release(self):
        self.device.release()

    def init_eye_detection(self, cascade_file: str):
        """
        Initialize eye detection

        :param model Haar cascade file
        """
        self.eye_detection = False
        self.cascade = cv.CascadeClassifier(cascade_file)
        self.eye_detection = True

    def update_frame(self) -> np.ndarray:
        """
        Update the eye frame based on eye detection
        """
        new_frame, frame = self.device.read()
        if new_frame and not np.sum(frame) == 0:
            self.new_frame = True
            self.frame = frame
        return(self.new_frame)

    def detect_eye(self) -> tuple:
        """
        Updates the position and size of the eye frame
        """
        self.eye_detected = False
        if self.new_frame:
            Eyes = self.cascade.detectMultiScale(
                self.frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)) ## <-- parametrize me
            if len(Eyes) == 1:
                self.eye_detected = True
                self.eye_frame_coords = Eyes[0]
        return(self.eye_detected)

    
    def update_eye_frame(self) -> np.ndarray:
        """
        Trims the frame obtained by eye detection
        """
        if self.new_frame:
            x, y, w, h = self.eye_frame_coords
            self.eye_frame = self.frame[y:y + h, x:x + w] 
            self.eye_frame = cv.cvtColor(self.eye_frame, cv.COLOR_BGR2GRAY)
        return(self.eye_frame)
    
    def update_quad_bright(self) -> tuple:
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
    
    def record_calib_data(self, target_pos: tuple) -> ary:
        """
        Record the present quad brightness for training the model

        :param target_pos (x, y) position of calibration target
        """
        new_data = np.append(self.quad_bright, ary(target_pos))
        self.calib_data = np.append(self.calib_data, [new_data], axis = 0)
        return(new_data)

    def train(self) -> lm.LinearRegression:
        """
        Trains the eye tracker
        """
        Quad = self.calib_data[:, 0:4]
        Pos = self.calib_data[:, 4:6]
        model = lm.LinearRegression()
        model.fit(Quad, Pos)
        self.model = model
        return(self.model)

    def update_offsets(self, target_pos: tuple) -> tuple:
        """
        Updates the offset values based on current eye_pos and a given target position

        param target_pos = position of visual target
        """
        # O = E - T
        # T = E - O
        new_offsets = ary(target_pos) - ary(self.eye_raw) # <---
        self.offsets = tuple(new_offsets)
        return(self.offsets)

    def reset_offsets(self) -> None:
        self.offsets = (0,0)


    def update_eye_pos(self) -> tuple:
        """
        Predicts the eye coordinates

        """        
        quad = ary(self.quad_bright)
        quad.shape = (1, 4)
        self.eye_raw = tuple(self.model.predict(quad)[0, :])
        # self.eye_pos = (x - self.offsets[0], y - self.offsets[1])
        self.eye_pos = tuple(ary(self.eye_raw) + ary(self.offsets))
        self.eye_pro = tuple(ary(self.eye_pos)/ary(self.surf_size))
        return self.eye_pos


    def update_eye_stim(self, Stim: Stimulus) -> tuple:
        """
        Returns the position relative to the stimulus
        """
        offsets = ary(Stim.pos)
        scale = ary(Stim.scale)
        # self.eye_stim = tuple((ary(self.eye_pos) - offsets)/scale)
        self.eye_stim = tuple((ary(self.eye_pos) - offsets)/scale)
        self.eye_pro = tuple(ary(self.eye_stim)/ary(Stim.size))
        return self.eye_stim



    def record(self, Exp_ID: str, Part_ID: str, Stim_ID: str) -> pd.DataFrame:
        """
        Records the eye coordinates

        """       
        
        new_data = pd.DataFrame({"Exp": Exp_ID,
                                    "Part": Part_ID,
                                    "Stim": Stim_ID, 
                                    "time" : time(),
                                    "x": self.eye_stim[0], 
                                    "y": self.eye_stim[1]}, 
                                  index = [0])
        new_data["x_pro"] = self.eye_pro[0]
        new_data["y_pro"] = self.eye_pro[1]
    
        self.data = pd.concat([self.data, new_data])
        return(new_data)


    def reset_calib(self) -> None:
        self.calib_data = np.zeros(shape=(0, 6))
        if hasattr(self, "model"):
            del self.model

    
    def reset_data(self) -> None:
        self.data = pd.DataFrame(columns = YETI.data_cols , 
                       dtype = "float64")

    def reset(self) -> None:
        self.reset_calib()
        self.reset_data()

    def draw_follow(self, surface: pg.Surface, add_raw = False, add_stim = False) -> None:
        """
        Draws a circle to the current eye position

        Note that eye positions must be updated using the update methods
        """ 
        surf_size = ary(surface.get_size())
        circ_size = int(surf_size.min()/50)
        circ_stroke = int(surf_size.min()/200)
        circle(surface, (255, 0, 0),  self.eye_pos, circ_size, circ_stroke)
        if add_raw:
            circle(surface, (0, 255, 0),  self.eye_raw, circ_size, circ_stroke)
        if add_stim:
            circle(surface, (0, 0, 255),  self.eye_stim, circ_size, circ_stroke)


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
    pro_positions : tuple[int]
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

    def __init__(self, surface: pg.Surface, 
                 pro_positions = (0.125, 0.5, 0.875)) -> None:
        self.surface = surface
        self.surface_size = ary(self.surface.get_size())
        self.pro_positions = ary(pro_positions)
        x_pos = self.pro_positions * self.surface_size[0]
        y_pos = self.pro_positions * self.surface_size[1]
        self.targets = ary(list(itertools.product(x_pos,y_pos))) ## No idea how this works
        self.active = 0
    
    def shuffle(self, reset = True):
        self.reset()
        self.targets = np.random.shuffle(self.targets)

    def active_pos(self) -> int:
        return self.targets[self.active]

    def reset(self) -> None:
        self.active = 0

    def n(self) -> int:
        return len(self.targets[:,0])

    def remaining(self) -> int:
        return self.n() - self.active - 1 

    def next(self) -> tuple:
        if self.remaining():
            this_target = self.targets[self.active]
            self.active += 1
            return True, this_target
        else:
            return False, None

    def draw(self) -> None:
        index = 0
        for target in self.targets:
            pos = list(map(int, target))
            if index == self.active:
                color = self.active_color
            else:
                color = self.color
            index += 1
            circle(self.surface, color, pos, self.radius, self.stroke)



