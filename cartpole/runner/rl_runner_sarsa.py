#!/usr/bin/env python
import numpy as np
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.state.tabular_qsa import tabular_qsa
from cartpole.env.cartpole_environment import cartpole_environment
from cartpole.misc.clear import clear
import time
import math
import h5py

class rl_runner_sarsa(object):
    def run_sim(self,p):
        #init random number generator from seed
        np.random.seed(p['random_seed']);
   
        #initialize hyperparameters fresh, unless we are resuming a saved simulation
        #in which case, we load the parameters
        if(not p.has_key('load_name')):
            self.init_hyperparameters(p)
        else:
            self.load_hyperparameters(p)


        #initialize environment
        self.sim = cartpole_environment()
        self.vel_bound = p['vel_bound']
        self.pos_bound = p['pos_bound']
        self.angle_vel_bound = p['angle_vel_bound']
        self.sim.init(self.vel_bound,self.angle_vel_bound,self.pos_bound,
            p['g'],p['l'],p['mp'],p['mc'],p['dt'],p['negative_reward'],p['positive_reward'],p['no_reward'])

        self.do_vis = p['do_vis']
        self.save_images = p.get('save_images',False)
        self.image_save_dir = p.get('image_save_dir',None)
        save_interval = p['save_interval']

        self.showevery = p['showevery']
        self.fastforwardskip = 5
        push_force = p['push_force']


        if(self.do_vis):
            #only import if we need it, since we don't want to require installation of pygame
            from cartpole.vis.visualize_sdl import visualize_sdl
            v = visualize_sdl()
            v.init_vis(p['display_width'],p['display_height'],p['axis_x_min'],p['axis_x_max'],p['axis_y_min'],p['axis_y_max'],p['fps'])

        print_update_timer = time.time()
        start_time = time.time()
        elapsed_time = time.time()
        step_duration_timer = time.time()
        save_time = time.time()
        avg_step_duration = 1.0

        ##repeat for each episode
        self.r_sum_avg = -10.0
        while 1:
            self.step = 0 
            ##initialize s
            self.sim.reset_state()
            self.s = self.sim.get_state()
            #choose a from s using policy derived from Q
            self.a = self.choose_action(self.s);

            self.r_sum = 0.0
            #repeat steps
            quit = False
            save_and_exit = False

            while 1:
                ##take action a, observe r, s'
                a_vel = [0.0,-push_force,push_force]
                self.sim.set_action(a_vel[self.a])

                self.sim.step()
                #print("Terminal: " + str(self.sim.is_terminal))
                self.r = self.sim.get_reward()
                self.s_prime = self.sim.get_state()
                self.r_sum += self.r

                #choose a' from s' using policy derived from Q
                self.a_prime = self.choose_action(self.s_prime)
                
                #Q(s,a) <- Q(s,a) + alpha[r + gamma*Q(s_prime,a_prime) - Q(s,a)]
                qsa_tmp = self.qsa.load(self.s,self.a)
                self.qsa.store(self.s,self.a,qsa_tmp +  \
                    self.alpha*(self.r + self.gamma*self.qsa.load(self.s_prime,self.a_prime) - qsa_tmp))
                
                
                if(self.do_vis):
                    if not (self.episode % self.showevery):
                        self.fast_forward = False
                        v.delay_vis()
                        v.draw_cartpole(self.sim.get_state(),self.a,self.sim.get_reward(),self)
                        exit = v.update_vis()
                        if(exit):
                            quit=True
                    elif(self.step == 0 and not (self.episode % self.fastforwardskip)):
                        self.fast_forward = True
                        v.delay_vis()
                        v.draw_cartpole(self.sim.get_state(),self.a,self.sim.get_reward(),self)
                        exit = v.update_vis()
                        if(exit):
                            quit=True
                        
                    #if(p.has_key('print_state_debug') and p['print_state_debug'] == True):
                    #    print("action: " + str(a) + " r: " + str(r) + \
                    #        " Qsa: " + str(self.qsa.load(s,a)) +  " state: " + str(s))
                    #    print("Qs0: " + str(self.qsa.load(s,0)))
                    #    print("Qs1: " + str(self.qsa.load(s,1)))
                    #    print("Qs2: " + str(self.qsa.load(s,2)))

                #TODO: put this printout stuff in a function
                if(print_update_timer < time.time() - 1.0):
                    clear()
                    print("Simname: " + str(p['simname']))
                    print("Episodes Elapsed: " + str(self.episode))
                    print("Average Reward Per Episode: " + str(self.r_sum_avg))
                    print("Epsilon: " + str(self.epsilon))
                    print("Average Steps Per Second: " + str(1.0/avg_step_duration))
                    m, s = divmod(time.time() - start_time, 60)
                    h, m = divmod(m, 60)
                    print "Elapsed Time %d:%02d:%02d" % (h, m, s)
                    print_update_timer = time.time()

                if(self.episode >= p['train_episodes']):
                    save_and_exit = True
                    quit=True

                if(quit):
                    break
                if(self.sim.is_terminal):
                    break
                if(self.step > 3000):
                    break
                ## s <- s';  a <-- a'
                self.s = self.s_prime
                self.a = self.a_prime

                self.step += 1
                avg_step_duration = 0.999*avg_step_duration + 0.001*(time.time() - step_duration_timer)
                step_duration_timer = time.time()
                #end step loop
            self.r_sum_avg = 0.9999*self.r_sum_avg + 0.0001*self.r_sum
            
            if(p.has_key('epsilon_decay')):
                self.epsilon = self.epsilon * p['epsilon_decay']
            if(p.has_key('epsilon_min')):
                self.epsilon = max(p['epsilon_min'],self.epsilon)

            #save stuff (TODO: Put this in a save function)
            if(time.time() - save_time > save_interval or save_and_exit == True):
                print('saving results...')
                self.save_results(p['results_dir'] + p['simname'] + p['version'] + '.h5py',p)
                save_time = time.time();

            if(quit==True or save_and_exit==True):
                break;
            self.episode += 1
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

    def save_results(self,filename,p):
        f_handle = h5py.File(filename,'w')
        f_handle['qsa_values'] = np.array(self.qsa.data);
        f_handle['state_min'] = np.array(self.state_min);
        f_handle['state_max'] = np.array(self.state_max);
        f_handle['state_size'] = np.array(self.state_size);
        f_handle['num_actions'] = np.array(self.num_actions);
        f_handle['epsilon'] = np.array(self.epsilon)
        f_handle['epsilon_decay'] = np.array(self.epsilon_decay)
        f_handle['epsilon_min'] = np.array(self.epsilon_min)
        f_handle['alpha'] = np.array(self.alpha)
        f_handle['gamma'] = np.array(self.gamma)
        f_handle['episode'] = np.array(self.episode)
        #TODO: save and load more hyperparameters, such as cart-pole dynamics, bounds, positive and negative reward values, and others?

        #iterate through all parameters and save them in the parameters group
        p_group = f_handle.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f_handle.close();

    def load_hyperparameters(self,p):
        f_handle = h5py.File(p['load_name'],'r')
        self.epsilon = f_handle['epsilon'].value
        self.epsilon_decay = f_handle['epsilon_decay'].value
        self.epsilon_min = f_handle['epsilon_min'].value
        self.alpha = f_handle['alpha'].value
        self.gamma = f_handle['gamma'].value
        self.state_min = list(f_handle['state_min'])
        self.state_max = list(f_handle['state_max'])
        self.state_size = list(f_handle['state_size'])
        self.episode = f_handle['episode'].value
        self.num_actions = 3
        self.qsa = tabular_qsa()
        self.qsa.init(self.state_min,self.state_max,self.state_size,self.num_actions)
        self.qsa.data = np.array(f_handle['qsa_values'])
        print('loaded epsilon: ' + str(self.epsilon))
        f_handle.close();

    def init_hyperparameters(self,p):
        self.epsilon = p['epsilon']
        self.epsilon_decay = p.get('epsilon_decay',1.0)
        self.epsilon_min = p.get('epsilon_min',self.epsilon)
        self.alpha = p['alpha']
        self.gamma = p['gamma']

        ##initialize Qsa arbitrarily
        vel_bound = p['vel_bound']
        pos_bound = p['pos_bound']
        angle_vel_bound = p['angle_vel_bound']
        self.state_min = [0.0, -vel_bound, -pos_bound, -angle_vel_bound]
        self.state_max = [2*math.pi,  vel_bound,  pos_bound,  angle_vel_bound]

        self.state_size = [p['angle_bins'],p['angle_vel_bins'],p['pos_bins'],p['vel_bins']]
        self.episode = 0

        self.num_actions = 3
        self.qsa = tabular_qsa()
        self.qsa.init(self.state_min,self.state_max,self.state_size,self.num_actions)

if __name__ == '__main__':
    g = rl_runner_sarsa()
    p = {}
    g.run_sim(p)
