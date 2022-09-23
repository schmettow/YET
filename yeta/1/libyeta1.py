import sys
import os
import time
import pandas as pd
import pygame as pg
import logging as log

def setup():
    """
    Creates global variables for Yeta_1 and changes the working directory
    """
    global WD, STIM_DIR, STIM_INFO
    global RESULT_DIR, PART_ID, RESULT_FILE
    global EYECASC
    global col_black, col_green, col_white
    global YETA, YETA_NAME
    global EXP_ID, SURF_SIZE

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
    RESULT_FILE = os.path.join(RESULT_DIR, YETA_NAME + "_" + EXP_ID + EXPERIMENTER + PART_ID + ".csv")
    """File name for data"""
    EYECASC = "haarcascade_eye.xml"

    ## Meta data of Yeta
    YETA = 1
    YETA_NAME = "Yeta" + str(YETA)

    ##### Logging #####
    log.basicConfig(filename='YET.log', level=log.INFO)

    # Colour definitions
    col_black = (0, 0, 0)
    col_green = (0, 255, 0)
    col_white = (255, 255, 255)

def init_pygame():
    # Pygame init
    pg.init()
    global FONT 
    global Font
    global font
    global SURF
    
    FONT = pg.font.Font('freesansbold.ttf', int(20 * min(SURF_SIZE) / 800))
    Font = pg.font.Font('freesansbold.ttf', int(15 * min(SURF_SIZE) / 800))
    font = pg.font.Font('freesansbold.ttf', int(12 * min(SURF_SIZE) / 800))
    pg.display.set_mode(SURF_SIZE)
    pg.display.set_caption(YETA_NAME)
    SURF = pg.display.get_surface()




class Stimulus:
  width = 800
  height = 800
  stim_dir = "Stimuli/"
  screen_size = (800, 800)

  def __init__(self):
    self.file = str()
    self.path = str()

  def __init__(self, row):
    row = row.to_dict()
    self.width = row["width"]
    self.height = row["height"]
    self.file = row["File"]
    self.path = os.path.join(self.stim_dir,  self.file)
    
  def load(self):
    try:
      image = pg.image.load(self.path)
    except:
      print("File at " + str(self.path)) + " could not be loaded."
    self.image = pg.transform.smoothscale(image, self.screen_size)
    
  def show(self, screen):
    screen.blit(self.image, (0, 0))


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

def main():
  setup()
  STIMS = StimulusSet(STIM_INFO)
  for stim in STIMS.Stimuli:
    print(stim.path)
