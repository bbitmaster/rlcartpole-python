#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import sys
from scipy import integrate
#import pylab as p

class pendulum_sim(object):
    def dx_dt(self,X,t=0):
        return np.array([X[1],
            -self._friction*X[1] -self._g*np.sin(X[0])])
    def __init__(self,g=9.81,l=1.0,friction=0.1):
        self._g = g
        self._l = l
        self._friction = friction

    #X[0] is angle
    #X[1] is initial velocity
    def generate_data(self,X0,t_start,t_end,num_points,include_angle=True,include_omega=True):
        t = np.linspace(t_start,t_end,num_points)

        X, infodict = integrate.odeint(self.dx_dt,X0,t,full_output=True)
        angle,omega = X.T

        #print("debug infodict['message']" + str(infodict['message']))

        x_pos = np.sin(angle)
        y_pos = -np.cos(angle)

        ret = {}
        if(include_angle):
            ret['angle'] = angle
        if(include_omega):
            ret['omega'] = omega
        ret['x_pos'] = x_pos
        ret['y_pos'] = y_pos
        ret['t'] = t
        return ret

class pendulum_plot_tools(object):
    def __init__(self):
        pass
    def plot_data(self,t,angle=None,omega=None,x_pos=None,y_pos=None):
        legend = []
        if(angle is not None):
            plt.plot(t,angle,'r-',label='Angle')
            legend.append("Angle")
        if(omega is not None):
            plt.plot(t,omega,'b-',label='Angular Velocity')
            legend.append("Angular Velocity")
        if(x_pos is not None):
            plt.plot(t,x_pos,'c-',label='X Position')
            legend.append("X Position")
        if(y_pos is not None):
            plt.plot(t,y_pos,'m-',label='Y Position')
            legend.append("Y Position")
        plt.grid()
        plt.xlabel('time')
        plt.legend(legend)
    #call this AFTER calling plot_data to display the plot
    def show_plot(self,):
        plt.show()

if(__name__ == '__main__'):
    if(len(sys.argv) < 2):
        print('usage: ./pendulum_tools.py <test_number> <test_parameters>')
    elif(sys.argv[1] == '1'):
        #Some very basic test code here. This is not a complete test!
        pend = pendulum_sim(friction=0.008)
        data = pend.generate_data([0.,30.],0.,1000.,60000)
        pend_plot = pendulum_plot_tools()
    
        pend_plot.plot_data(data['t'],x_pos=data['x_pos'],y_pos=data['y_pos'])
        pend_plot.show_plot()
    else:
        print("no valid test entered")
