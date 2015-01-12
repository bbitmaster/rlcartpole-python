#!/usr/bin/env python
import pygame
from cartpolesim import cartpole_sim
import math
#from pygame.locals import *
SCREEN_WIDTH=1024
SCREEN_HEIGHT=768

AXIS = [-5,5,-5,5]


class cartpole_game(object):
    def init_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        pygame.display.set_caption("Cart Pole Display")

        pygame.display.flip()
        self.cp_sim = cartpole_sim()

    def run_game(self):
        clock = pygame.time.Clock()
        while 1:
            clock.tick(60)
            self.screen.fill((0,0,0))
            self.draw_cartpole()

            pygame.display.flip()
            self.cp_sim.step()
            if(self.cp_sim.state[2] < -4.0):
                self.cp_sim.state[2] = -4.0
            if(self.cp_sim.state[2] > 4.0):
                self.cp_sim.state[2] = 4.0
                
            pressed = pygame.key.get_pressed()
            self.cp_sim.u = 0.0
            if(pressed[pygame.K_LEFT]):
                self.cp_sim.u = -10.0
            if(pressed[pygame.K_RIGHT]):
                self.cp_sim.u = 10.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            print(str(self.cp_sim.u))

    #convert x,y coords from the axis scale to screen coordinates
    def convert_coords(self,coords):
        self.axis = AXIS
        img_coords = [0,0]
        img_width = SCREEN_WIDTH
        img_height = SCREEN_HEIGHT
        img_coords[0] = float(((coords[0] - self.axis[0])/float(self.axis[1] - self.axis[0]))*float(img_width))
        img_coords[1] = float(((coords[1] - self.axis[2])/float(self.axis[3] - self.axis[2]))*float(img_height))
        return tuple(img_coords);

    #convert x,y,width,height coords from the axis scale to screen coordinates
    def convert_rect_coords(self,coords):
        self.axis = AXIS
        img_coords = [0,0,0,0]
        img_width = SCREEN_WIDTH
        img_height = SCREEN_HEIGHT
        img_coords[0] = float(((coords[0] - self.axis[0])/float(self.axis[1] - self.axis[0]))*float(img_width))
        img_coords[1] = float(((coords[1] - self.axis[2])/float(self.axis[3] - self.axis[2]))*float(img_height))
        img_coords[2] = float(((coords[2])/float(self.axis[1] - self.axis[0]))*float(img_width))
        img_coords[3] = float(((coords[3])/float(self.axis[3] - self.axis[2]))*float(img_height))
        return tuple(img_coords);

    def draw_cartpole(self):
        #draw cart
        angle = self.cp_sim.state[0]
        pos = self.cp_sim.state[2]
        coords = self.convert_rect_coords((pos - 0.25,-0.25,0.5,0.5))
        self.screen.fill((192,24,24),pygame.Rect(coords))
        
        #draw pole
        pole_coords1 = self.convert_coords((pos,0.0))
        pole_coords2 = self.convert_coords((pos + 2.0*math.sin(angle),-2.0*math.cos(angle)))
        pygame.draw.line(self.screen,(24,24,192),pole_coords1,pole_coords2,4)


if __name__ == '__main__':
    g = cartpole_game()
    g.init_game()
    g.run_game()
