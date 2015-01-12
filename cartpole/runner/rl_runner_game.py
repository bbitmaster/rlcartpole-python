#!/usr/bin/env python
import pygame
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.env.cartpole_environment import cartpole_environment
from cartpole.vis.visualize_sdl import visualize_sdl


import math
#from pygame.locals import *
SCREEN_WIDTH=1024
SCREEN_HEIGHT=768

AXIS = [-5,5,-5,5]


vel_bound = 15;
angle_vel_bound = 15;
pos_bound = 10

class game_runner(object):
    #TODO: Parameters for displaying to screen, and printout, and etc

    def run_sim(self,parameters):
        sim = cartpole_environment()
        sim.init(vel_bound,angle_vel_bound,pos_bound)
        v = visualize_sdl()
        v.init_vis()
        v.delay = 30;

        while 1:
            v.delay_vis()
            k = v.get_keys()

            u = 0.0;
            if(k[0]):
                u = -10.0;
            if(k[1]):
                u = 10.0;
            sim.set_action(u)
            sim.step()
            #if(sim.state[2] < -4.0):
            #    sim.state[2] = -4.0
            #if(sim.state[2] > 4.0):
            #    sim.state[2] = 4.0
            if(sim.is_terminal):
                sim.reset_state()
                
            v.draw_cartpole(sim.get_state(),sim.get_reward())
            exit = v.update_vis()
            if(exit):
                break
        return



if __name__ == '__main__':
    g = game_runner()
    p = {}
    g.run_sim(p)
