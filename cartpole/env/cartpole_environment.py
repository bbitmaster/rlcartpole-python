#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import sys
from matplotlib.mlab import rk4
from scipy import integrate
from math import sin, cos, pi
from cartpole.sim.cartpole_sim import cartpole_sim

class cartpole_environment(object):
    def init(self,vel_bound,angle_vel_bound,pos_bound,g=9.81,l=0.5,mp=0.01,mc=1.0,dt=0.02):
        self.sim = cartpole_sim(g,l,mp,mc,dt)
        self.vel_bound = vel_bound
        self.angle_vel_bound = angle_vel_bound
        self.pos_bound = pos_bound
        self.is_terminal = False

        self.negative_reward = -10.0
        self.positive_reward = 0.1
        self.no_reward = 0.0

    #TODO: generate random state
    def reset_state(self):
        self.sim.init_state(2.0,0.0,0.0,0.0)

    def set_action(self,u):
        self.sim.u = u

    def step(self):
        self.sim.step()

        #test if we have an out of bounds state
        self.is_terminal = False
        if(abs(self.sim.state[1]) > self.angle_vel_bound):
            self.is_terminal = True
        if(abs(self.sim.state[2]) > self.pos_bound):
            self.is_terminal = True
        if(abs(self.sim.state[3]) > self.vel_bound):
            self.is_terminal = True

        return self.is_terminal

    def get_state(self):
        return self.sim.state

    def get_reward(self):
        if(self.is_terminal):
            return self.negative_reward
        angle = self.sim.state[0]

        if(angle < 2*pi/20 or angle > (2*pi - 2*pi/20)):
            return self.positive_reward

        return self.no_reward
            


