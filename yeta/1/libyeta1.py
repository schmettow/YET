import sys
import os
import time
import numpy as np
import pandas as pd
import pygame as pg
import logging as log


class Stimulus:
  stim_dir = "Stimuli/"

  def __init__(self, entry):
    if isinstance(entry, pd.DataFrame):
        entry = entry.to_dict()
    self.file = entry["File"]
    self.path = os.path.join(self.stim_dir,  self.file)
    self.size = np.array((entry["width"], entry["height"]))
    
  def load(self, surface: pg.Surface, scale = True):
    image = pg.image.load(self.path)
    self.surface = surface
    self.surf_size = np.array(self.surface.get_size())
    if scale:
        self.scale = min(self.surf_size / self.size)
        scale_to = np.array(self.size * self.scale).astype(int)
        self.image = pg.transform.smoothscale(image, scale_to)    
        self.size = self.image.get_size()
    else:
        self.scale = 1
    self.pos = np.array((self.surf_size - self.size)/2).astype(int)

  def draw(self):
    self.surface.blit(self.image, self.pos)


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
