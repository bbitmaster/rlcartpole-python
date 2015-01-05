#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import sys
from matplotlib.mlab import rk4
from scipy import integrate
from math import sin, cos
#import pylab as p

class cartpole_sim(object):
    def __init__(self,g=9.81,l=0.5,mp=0.1,mc=1.0,dt=0.02):
        self.g = g
        self.l = l
        self.mp = mp
        self.mc = mc
        self.dt = dt
        self.u = 0.0
        self.state = (2.0,0.0,0.0,0.0)

    def init_state(angle,angle_deriv,x,x_deriv):
        self.state = (angle,angle_deriv,x,x_deriv)

    def _derivs(self,x,t):
        F = self.u
        (theta,theta_,_s,s_) = x
        u = theta_
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        mp = self.mp
        mc = self.mc
        l = self.l
        u_ = (self.g * sin_theta * (mc + mp) - (F + mp * l * theta ** 2 * sin_theta) * cos_theta) / (4 / 3 * l * (mc + mp) - mp * l * cos_theta ** 2)
        v = s_
        v_ = (F - mp * l * (u_ * cos_theta - (s_ ** 2 * sin_theta))) / (mc + mp)
        return (u, u_, v, v_)

    def step(self):
        self.state = rk4(self._derivs,self.state,[0,self.dt])
        self.state = self.state[-1]
        print("state: " + str(self.state))

