#!/usr/bin/env python
import pygame
import numpy as np
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.vis.visualize_sdl import visualize_sdl
from cartpole.state.tabular_qsa import tabular_qsa
from cartpole.env.cartpole_environment import cartpole_environment

import math
#from pygame.locals import *

SCREEN_WIDTH=1024
SCREEN_HEIGHT=768

vel_bound = 15;
angle_vel_bound = 13;
pos_bound = 20

class rl_runner_sarsa(object):
    def run_sim(self,parameters):
        #should be a hyperparameter
        self.epsilon = 0.10
        #initialize environment
        self.sim = cartpole_environment()
        self.sim.init(vel_bound,angle_vel_bound,pos_bound)

        ##initialize Qsa arbitrarily

        state_min = [0.0, -vel_bound-1.0, -pos_bound-2.0, -angle_vel_bound -4.0]
        state_max = [2*math.pi,  vel_bound+1.0,  pos_bound+2.0,  angle_vel_bound + 4.0]
        state_size = [20,20,15,20]
        self.num_actions = 3

        self.qsa = tabular_qsa()
        self.qsa.init(state_min,state_max,state_size,self.num_actions)
        self.alpha = 0.8
        self.gamma = 0.8
        self.do_vis = True
        self.frameskip = 2
        self.showevery = 300

        if(self.do_vis):
            v = visualize_sdl()
            v.init_vis()
        episode = 0

        ##repeat for each episode
        r_sum_avg = -10.0
        while 1:
            step = 0 
            ##initialize s
            self.sim.reset_state()
            s = self.sim.get_state()
            #choose a from s using policy derived from Q
            a = self.choose_action(s);

            r_sum = 0.0
            #repeat steps
            quit = False
            if(episode > 10000):
                self.epsilon = .01
            if(episode > 100000):
                self.epsilon = .001
            if(episode > 200000):
                self.epsilon = .0001
            while 1:
                ##take action a, observe r, s'
                a_vel = [0.0,-10.0,10.0]
                self.sim.set_action(a_vel[a])

                self.sim.step()
                #print("Terminal: " + str(self.sim.is_terminal))
                r = self.sim.get_reward()
                s_prime = self.sim.get_state()
                r_sum += r

                #choose a' from s' using policy derived from Q
                a_prime = self.choose_action(s_prime)
                
                #Q(s,a) <- Q(s,a) + alpha[r + gamma*Q(s_prime,a_prime) - Q(s,a)]
                qsa_tmp = self.qsa.load(s,a)
                self.qsa.store(s,a,qsa_tmp +  \
                    self.alpha*(r + self.gamma*self.qsa.load(s_prime,a_prime) - qsa_tmp))
                
                if(not episode % self.showevery):
                    print("action: " + str(a) + " r: " + str(r) + \
                        " Qsa: " + str(self.qsa.load(s,a)) +  " state: " + str(s))
                    print("Qs0: " + str(self.qsa.load(s,0)))
                    print("Qs1: " + str(self.qsa.load(s,1)))
                    print("Qs2: " + str(self.qsa.load(s,2)))
                
                if(self.do_vis and not (step % self.frameskip) and not (episode % self.showevery)):
                    v.delay_vis()
                    v.draw_cartpole(self.sim.get_state(),self.sim.get_reward())
                    exit = v.update_vis()
                    if(exit):
                        quit=True
                if(quit):
                    break
                if(self.sim.is_terminal):
                    break
                if(step > 3000):
                    break
                ## s <- s';  a <-- a'
                s = s_prime
                a = a_prime

                step += 1
                #end step loop
            r_sum_avg = 0.9999*r_sum_avg + 0.0001*r_sum
            print("episode: " + str(episode) + " steps: " + str(step) + " r_sum: " + str(r_sum) + " avg: " + str(r_sum_avg))

            if(quit):
                break;
            episode += 1
            #end episode loop
        return

    def choose_action(self,state):
        max_action = -1e99

        #epsilon-greedy
        if(np.random.random() < self.epsilon):
            a = np.random.randint(self.num_actions)
        else:
            a = np.argmax(np.array([self.qsa.load(state,i) for i in range(self.num_actions)]))

        return a




if __name__ == '__main__':
    g = rl_runner_sarsa()
    p = {}
    g.run_sim(p)