#!/usr/bin/env python
import numpy as np
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.state.tabular_qsa import tabular_qsa
from cartpole.env.cartpole_environment import cartpole_environment

import math

class rl_runner_sarsa(object):
    def run_sim(self,p):
        #should be a hyperparameter
        self.epsilon = p['epsilon']
        #initialize environment
        self.sim = cartpole_environment()
        vel_bound = p['vel_bound']
        pos_bound = p['pos_bound']
        angle_vel_bound = p['angle_vel_bound']
        self.sim.init(vel_bound,angle_vel_bound,pos_bound,p['g'],p['l'],p['mp'],p['mc'],p['dt'],p['negative_reward'],p['positive_reward'],p['no_reward'])

        ##initialize Qsa arbitrarily
        #state_min = [0.0, -vel_bound-1.0, -pos_bound-2.0, -angle_vel_bound -4.0]
        #state_max = [2*math.pi,  vel_bound+1.0,  pos_bound+2.0,  angle_vel_bound + 4.0]
        state_min = [0.0, -vel_bound, -pos_bound, -angle_vel_bound]
        state_max = [2*math.pi,  vel_bound,  pos_bound,  angle_vel_bound]

        state_size = [p['angle_bins'],p['angle_vel_bins'],p['pos_bins'],p['vel_bins']]
        self.num_actions = 3

        self.qsa = tabular_qsa()
        self.qsa.init(state_min,state_max,state_size,self.num_actions)
        self.alpha = p['alpha']
        self.gamma = p['gamma']
        self.do_vis = p['do_vis']
        self.showevery = p['showevery']
        push_force = p['push_force']

        if(self.do_vis):
            #only import if we need it, since we don't want to require installation of pygame
            from cartpole.vis.visualize_sdl import visualize_sdl
            v = visualize_sdl()
            v.init_vis(p['display_width'],p['display_height'],p['axis_x_min'],p['axis_x_max'],p['axis_y_min'],p['axis_y_max'],p['fps'])
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
            while 1:
                ##take action a, observe r, s'
                a_vel = [0.0,-push_force,push_force]
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
                
                
                if(self.do_vis and not (episode % self.showevery)):
                    v.delay_vis()
                    v.draw_cartpole(self.sim.get_state(),self.sim.get_reward())
                    exit = v.update_vis()
                    if(exit):
                        quit=True
                    if(p.has_key('print_state_debug') and p['print_state_debug'] == True):
                        print("action: " + str(a) + " r: " + str(r) + \
                            " Qsa: " + str(self.qsa.load(s,a)) +  " state: " + str(s))
                        print("Qs0: " + str(self.qsa.load(s,0)))
                        print("Qs1: " + str(self.qsa.load(s,1)))
                        print("Qs2: " + str(self.qsa.load(s,2)))

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
            
            if(p.has_key('epsilon_decay')):
                epsilon = epsilon * p['epsilon_decay']
            if(p.has_key('epsilon_min')):
                epsilon = min(p['epsilon_min'],epsilon)

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
