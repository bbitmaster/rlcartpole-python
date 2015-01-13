#!/usr/bin/env python
import pygame
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.env.cartpole_environment import cartpole_environment
from cartpole.vis.visualize_sdl import visualize_sdl


import math
import curses

class runner_game(object):
    #TODO: Parameters for displaying to screen, and printout, and etc

    def run_sim(self,p):
        sim = cartpole_environment()
        sim.init(p['vel_bound'],p['angle_vel_bound'],p['pos_bound'],p['g'],p['l'],p['mp'],p['mc'],p['dt'],p['negative_reward'],p['positive_reward'],p['no_reward'])
        v = visualize_sdl()
        v.init_vis(p['display_width'],p['display_height'],p['axis_x_min'],p['axis_x_max'],p['axis_y_min'],p['axis_y_max'],p['fps'])
        push_force = p['push_force']

        while 1:
            curses.initscr()
            if(p.has_key('print_state_debug')):
                print(str(sim.sim.state))
            v.delay_vis()
            k = v.get_keys()

            u = 0.0;
            if(k[0]):
                u = -push_force;
            if(k[1]):
                u = push_force;
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
    g = runner_game()
    p = {}
    g.run_sim(p)
