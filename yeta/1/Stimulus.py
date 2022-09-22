import sys
import os
import pandas as pd
import pygame as pg

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
  WD = os.path.dirname(sys.argv[0])
  os.chdir(WD)
  STIM_PATH = os.path.join(WD, "Stimuli")
  """Directory where stimuli reside"""
  STIM_INFO = os.path.join(STIM_PATH, "Stimuli.csv")
  """CSV file describing stimuli"""
  STIMS = StimulusSet(STIM_INFO)
  for stim in STIMS.Stimuli:
    print(stim.path)

#main()


