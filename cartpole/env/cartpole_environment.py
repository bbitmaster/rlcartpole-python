#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import sys
from scipy import integrate
from math import sin, cos, pi
from cartpole.sim.cartpole_sim import cartpole_sim

class cartpole_environment(object):
    def init(self,vel_bound,angle_vel_bound,pos_bound,g=9.81,l=0.5,mp=0.01,mc=1.0,dt=0.02,negative_reward=-10.0,positive_reward=0.1,no_reward=0.0,reward_type=0):
        self.sim = cartpole_sim(g,l,mp,mc,dt)
        self.vel_bound = vel_bound
        self.angle_vel_bound = angle_vel_bound
        self.pos_bound = pos_bound
        self.is_terminal = False

        self.negative_reward = negative_reward
        self.positive_reward = positive_reward
        self.no_reward = no_reward

    #TODO: generate random state
    def reset_state(self):
        self.sim.init_state(np.random.random()*2*pi,(np.random.random() - 0.5)*self.angle_vel_bound/2.0,0.0,(np.random.random() - 0.5)*self.vel_bound/2.0)

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

    def get_reward(self,reward_type):
        if(self.is_terminal):
            return self.negative_reward
        angle = self.sim.state[0]
        angle_vel = self.sim.state[1]

        if(reward_type == 0):
            if(angle < 2*pi/20 or angle > (2*pi - 2*pi/20)):
                return self.positive_reward

        if(reward_type == 1):
            if(np.abs(angle_vel) < 4.0):
                if(angle < 2*pi/20 or angle > (2*pi - 2*pi/20)):
                    return self.positive_reward

        if(reward_type == 2):
            if(np.abs(angle_vel) < 5.0):
                if(angle < 2*pi/20 or angle > (2*pi - 2*pi/20)):
                    return self.positive_reward

        if(reward_type == 3):
            if(np.abs(angle_vel) < 6.0):
                if(angle < 2*pi/20 or angle > (2*pi - 2*pi/20)):
                    return self.positive_reward

        if(reward_type == 4):
            if(np.abs(angle_vel) < 5.0):
                if(angle < 2*pi/20 or angle > (2*pi - 2*pi/20)):
                    return self.positive_reward
            if(np.abs(angle_vel) < 6.0):
                if(angle < 2*pi/10 or angle > (2*pi - 2*pi/10)):
                    return self.positive_reward/8.0

        if(reward_type == 5):
            if(np.abs(angle_vel) < 5.0):
                if(angle < 2*pi/20 or angle > (2*pi - 2*pi/20)):
                    return self.positive_reward
            if(np.abs(angle_vel) < 6.0):
                if(angle < 2*pi/10 or angle > (2*pi - 2*pi/10)):
                    return self.positive_reward/8.0
            if(np.abs(angle_vel) < 6.0):
                if(angle < 2*pi/5 or angle > (2*pi - 2*pi/5)):
                    return self.positive_reward/16.0

        return self.no_reward
            


