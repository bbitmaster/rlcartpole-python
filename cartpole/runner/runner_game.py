#!/usr/bin/env python
import pygame
import numpy as np
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.env.cartpole_environment import cartpole_environment
from cartpole.vis.visualize_sdl import visualize_sdl
from cartpole.misc.clear import clear
import sys

import math
#import curses

class runner_game(object):
    #TODO: Parameters for displaying to screen, and printout, and etc
        
    def run_sim(self,p):
        sim = cartpole_environment()
        sim.init(p['vel_bound'],p['angle_vel_bound'],p['pos_bound'],p['g'],p['l'],p['mp'],p['mc'],p['dt'],p['negative_reward'],p['positive_reward'],p['no_reward'])
        v = visualize_sdl()
        v.init_vis(p['display_width'],p['display_height'],p['axis_x_min'],p['axis_x_max'],p['axis_y_min'],p['axis_y_max'],p['fps'])
        push_force = p['push_force']

        self.vel_bound = p['vel_bound']
        self.pos_bound = p['pos_bound']
        self.angle_vel_bound = p['angle_vel_bound']
        self.mins = np.array([0.0, -self.vel_bound, -self.pos_bound, -self.angle_vel_bound])
        self.maxs = np.array([2*math.pi,  self.vel_bound,  self.pos_bound,  self.angle_vel_bound])
        self.mins = np.append(np.array([-1.0,-1.0]),self.mins[1:])
        self.maxs = np.append(np.array([1.0,1.0]),self.maxs[1:])
        self.incorrect_target = p['incorrect_target']
        self.correct_target = p['correct_target']
        self.num_actions = 3
        action=0

        while 1:
            if(p.has_key('print_state_debug')):
                clear()
                action_list = np.ones((1,self.num_actions))*self.incorrect_target
                action_list[0,action] = self.correct_target
                state = sim.sim.state
                s = np.append(np.array([math.sin(state[0]),math.cos(state[0])]),state[1:])
                s = (np.array(s) - self.mins)/(self.maxs - self.mins)
                s = s-0.5
                s = s*2.25
                s = np.append(s,action_list)
                np.set_printoptions(precision=4)
                print(str(s[:,np.newaxis]))
                print(str(np.array(sim.sim.state)[:,np.newaxis]))
            v.delay_vis()
            k = v.get_keys()

            u = 0.0;
            action = 0
            if(k[0]):
                action = 2
                u = -push_force;
            if(k[1]):
                action = 1
                u = push_force;
            sim.set_action(u)
            sim.step()
            #if(sim.state[2] < -4.0):
            #    sim.state[2] = -4.0
            #if(sim.state[2] > 4.0):
            #    sim.state[2] = 4.0
            if(sim.is_terminal):
                sim.reset_state()

            v.draw_cartpole(sim.get_state(),action,sim.get_reward())
            exit = v.update_vis()
            if(exit):
                break
        return



if __name__ == '__main__':
    g = runner_game()
    p = {}
    g.run_sim(p)
