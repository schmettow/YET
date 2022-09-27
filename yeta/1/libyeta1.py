import numpy as np
import pygame as pg

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
