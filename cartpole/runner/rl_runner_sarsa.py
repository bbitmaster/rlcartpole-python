#!/usr/bin/env python
import time
import math
import sys
import numpy as np
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.state.tabular_qsa import tabular_qsa
from cartpole.state.nnet_qsa import nnet_qsa
from cartpole.state.cluster_nnet_qsa import cluster_nnet_qsa
from cartpole.state.cartpole_nnet_qsa import cartpole_nnet_qsa
from cartpole.state.recurrent_cartpole_nnet_qsa import recurrent_cartpole_nnet_qsa
from cartpole.state.cartpole_state_transformer import cartpole_state_transformer
from cartpole.env.cartpole_environment import cartpole_environment
from cartpole.misc.clear import clear
from cartpole.misc.save_h5py import save_results,load_results

class rl_runner_sarsa(object):
    def run_sim(self,p):
        #init random number generator from seed
        np.random.seed(p['random_seed']);
   
        #initialize hyperparameters fresh, unless we are resuming a saved simulation
        #in which case, we load the parameters
        if(not p.has_key('load_name')):
            self.init_sim(p)
        else:
            self.load_sim(p)

        #initialize environment
        self.sim = cartpole_environment()
        self.vel_bound = p['vel_bound']
        self.pos_bound = p['pos_bound']
        self.angle_vel_bound = p['angle_vel_bound']
        self.sim.init(self.vel_bound,self.angle_vel_bound,self.pos_bound,
            p['g'],p['l'],p['mp'],p['mc'],p['dt'],p['negative_reward'],p['positive_reward'],p['no_reward'],p.get('create_pomdp',False))

        self.do_vis = p['do_vis']
        self.save_images = p.get('save_images',False)
        self.image_save_dir = p.get('image_save_dir',None)
        save_interval = p['save_interval']
        self.do_running_printout = p.get('do_running_printout',False)

        self.showevery = p['showevery']
        self.fastforwardskip = 5
        push_force = p['push_force']


        self.do_recurrence = False
        if(p['do_recurrence']):
            self.do_recurrence = True

        if(self.do_vis):
            #only import if we need it, since we don't want to require installation of pygame
            from cartpole.vis.visualize_sdl import visualize_sdl
            v = visualize_sdl()
            v.init_vis(p['display_width'],p['display_height'],p['axis_x_min'],p['axis_x_max'],p['axis_y_min'],p['axis_y_max'],p['fps'])

        print_update_timer = time.time()
        self.start_time = time.time()
        elapsed_time = time.time()
        step_duration_timer = time.time()
        save_time = time.time()
        self.avg_step_duration = 1.0

        ##repeat for each episode
        self.r_sum_avg = -0.95
        self.r_sum_avg_list = []
        self.steps_balancing_pole_list = []
        self.steps_balancing_pole_avg = 0.00
        self.steps_balancing_pole_avg_list = []

        while 1:
            #reset eligibility at the beginning of each episode
            #TODO: This should be abstracted into a function call
            if(hasattr(self.qsa,'_lambda')):
                for l in self.qsa.net.layer:
                    l.eligibility = np.zeros(l.eligibility.shape,dtype=np.float32)

            self.step = 0 
            ##initialize s
            self.sim.reset_state()
            self.s = self.state_transformer.transform(self.sim.get_state())

            if(self.do_recurrence):
                #choose a from s using policy derived from Q
                self.h = np.zeros(p['num_hidden'],dtype=np.float32)
                (self.a,self.qsa_tmp,self.h_prime) = self.choose_action_recurrence(np.append(self.s,self.h),p);
            else:
                #choose a from s using policy derived from Q
                (self.a,self.qsa_tmp) = self.choose_action(self.s,p);

            r_list = []
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
                self.s_prime = self.state_transformer.transform(self.sim.get_state())
                self.r_sum += self.r
                r_list.append(self.r)

                if(self.do_recurrence):
                    (self.a_prime,self.qsa_prime,self.h_primeprime) = \
                            self.choose_action_recurrence(np.append(self.s_prime,self.h_prime),p)

                    current_s = np.append(self.s,self.h)
                    next_s = np.append(self.s_prime,self.h_prime)
                    self.qsa.store(current_s,self.a,self.qsa_tmp +  \
                        self.alpha*(self.r + self.gamma*self.qsa.load(next_s,self.a_prime) - self.qsa_tmp))
                else:
                    #choose a' from s' using policy derived from Q
                    (self.a_prime,self.qsa_prime) = self.choose_action(self.s_prime,p)
                
                    #Q(s,a) <- Q(s,a) + alpha[r + gamma*Q(s_prime,a_prime) - Q(s,a)]
                    #todo: qsa_prime can be saved and reused for qsa_tmp
                    #qsa_tmp = self.qsa.load(self.s,self.a)
                    #self.qsa.update(self.s,self.a,self.r,self.s_prime,self.a_prime,self.qsa_tmp)
                    self.qsa.store(self.s,self.a,self.qsa_tmp +  \
                    self.alpha*(self.r + self.gamma*self.qsa.load(self.s_prime,self.a_prime) - self.qsa_tmp))
                
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
                #the self.episode > 0 check prevents a bug where some of the printouts are empty arrays before the first episode completes
                if(self.do_running_printout and print_update_timer < time.time() - 1.0 and self.episode > 0):
                    self.do_running_printout()

                if(self.episode >= p['train_episodes']):
                    save_and_exit = True
                    quit=True

                if(quit):
                    break
                if(self.sim.is_terminal):
                    break
                if(self.step > p['max_steps']):
                    break
                ## s <- s';  a <-- a'
                self.s = self.s_prime
                self.a = self.a_prime
                self.qsa_tmp = self.qsa_prime
                if(self.do_recurrence):
                    self.h = self.h_prime
                    self.h_prime = self.h_primeprime

                #print("Next Step \n")
                self.step += 1
                self.avg_step_duration = 0.995*self.avg_step_duration + (1.0 - 0.995)*(time.time() - step_duration_timer)
                step_duration_timer = time.time()
                #end step loop

            #compute the number of steps that have a positive reward, as the number of steps that balanced
            self.steps_balancing_pole = np.sum(np.array(r_list) > 0.0000001)
            self.steps_balancing_pole_list.append(self.steps_balancing_pole)

            self.steps_balancing_pole_avg = 0.995*self.steps_balancing_pole_avg + (1.0 - 0.995)*self.steps_balancing_pole
            self.steps_balancing_pole_avg_list.append(self.steps_balancing_pole_avg)

            self.r_sum_avg = 0.995*self.r_sum_avg + (1.0 - 0.995)*self.r_sum
            
            if(p['decay_type'] == 'geometric'):
                self.epsilon = self.epsilon * p['epsilon_decay']
                self.epsilon = max(p['epsilon_min'],self.epsilon)
            elif(p['decay_type'] == 'linear'):
                self.epsilon = self.epsilon - p['epsilon_decay']
                self.epsilon = max(p['epsilon_min'],self.epsilon)
            

            if(p.has_key('learning_rate_decay_type') and p['learning_rate_decay_type'] == 'geometric'):
                self.alpha = self.alpha * p['learning_rate_decay']
                self.alpha = max(p['learning_rate_min']/p['learning_rate'],self.alpha)
            elif(p.has_key('learning_rate_decay_type') and p['learning_rate_decay_type'] == 'linear'):
                self.alpha = self.alpha - p['learning_rate_decay']
                self.alpha = max(p['learning_rate_min']/p['learning_rate'],self.alpha)

            #print debug for episode
            m, s = divmod(time.time() - self.start_time, 60)
            h, m = divmod(m, 60)
            sys.stdout.write(("ep: %d" % self.episode) + (" epsilon: %2.4f" %self.epsilon) + (" avg steps balanced: %2.4f" % self.steps_balancing_pole_avg) + (" max steps balanced: %2.4f" % np.max(np.array(self.steps_balancing_pole_avg_list))) + (" total_steps: %d" % self.step) + (" steps/sec: %2.4f" % (1.0/self.avg_step_duration)))
            if(p.has_key('zeta_decay') and p['zeta_decay'] is not None):
                sys.stdout.write(" zeta: %2.4f" % self.qsa.net.layer[0].zeta)
            sys.stdout.write(" l_rate: %2.4f" % (self.alpha*p['learning_rate']))
            print(" Time %d:%02d:%02d" % (h, m, s))

            #save stuff (TODO: Put this in a save function)
            if(time.time() - save_time > save_interval or save_and_exit == True):
                print('saving results...')
                self.save_results(p['results_dir'] + p['simname'] + p['version'] + '.h5py',p)
                save_time = time.time();

            if(quit==True or save_and_exit==True):
                break;
            self.episode += 1
            #end episode loop

        self.update_results(p)
        obj = np.max(self.results['steps_balancing_pole_avg_list'])
        argmax = np.argmax(self.results['steps_balancing_pole_avg_list'])
        print("obj: " + str(obj) + " argmax: " + str(argmax))
        return self.results

    def do_running_printout(self):
        clear()
        print("Simname: " + str(p['simname']))
        print("Episodes Elapsed: " + str(self.episode))
        print("Average Reward Per Episode: " + str(self.r_sum_avg))
        print("Average Number of Steps Spent Balancing Pole: " + str(self.steps_balancing_pole_avg))
        print("Max Number of Steps Spent Balancing Pole: " + str(np.max(np.array(self.steps_balancing_pole_avg_list))))
        print("Epsilon: " + str(self.epsilon))
        print("Epsilon Min: " + str(p['epsilon_min']))
        print("Alpha (learning rate): " + str(self.alpha*p['learning_rate']))
        if(p.has_key('learning_rate_decay')):
            print("Alpha (learning rate) decay: " + str(p['learning_rate_decay']))
        if(p['qsa_type'] == 'cluster_nnet'):
            print("num_hidden: " + str(p['num_hidden']))
            print('num_selected: ' + str(self.qsa.net.layer[0].num_selected))
        if(p['qsa_type'] == 'nnet'):
            print("Activation function: " + str(p['activation_function']))
            print("num_hidden: " + str(p['num_hidden']))
        if(p['action_type'] == 'noisy_qsa'):
            print("Average QSA Standard Deviation: " + str(self.qsa_std_avg))
            print("Probability of taking different action: " + str(self.prob_of_different_action))
        if(p['qsa_type'] == 'cartpole_nnet'):
            print("state given to nnet:\n" + str(np.array(self.qsa.net.input).transpose()))
        print("Average Steps Per Second: " + str(1.0/self.avg_step_duration))
        print("Action Type: " + str(p['action_type']))
        print("a_list: " + str(self.tmp_a_list))
        m, s = divmod(time.time() - self.start_time, 60)
        h, m = divmod(m, 60)
        print "Elapsed Time %d:%02d:%02d" % (h, m, s)
        sys.stdout.flush()
        print_update_timer = time.time()





    def choose_action(self,state,p):
        max_action = -1e99
        
        #epsilon-greedy
        if(p['action_type'] == 'e_greedy'):
            qsa_list = [self.qsa.load(state,i) for i in range(self.num_actions)]
            if(np.random.random() < self.epsilon):
                a = np.random.randint(self.num_actions)
                #print("selected action " + str(a) + "which had QSA value of: " + str(qsa_list[a]))
            else:
                a = np.argmax(np.array(qsa_list))
                #print("selected random action " + str(a) + "which had QSA value of: " + str(qsa_list[a]))
        elif(p['action_type'] == 'noisy_qsa'):
            #INIT CODE HERE
            if(self.step == 0 and self.episode == 0):
                self.qsa_std_avg = p['qsa_avg_init']
                self.qsa_avg_alpha = p['qsa_avg_alpha']
                #this will give a moving average estimate of the probability of selecting a different action
                #(used for printing only)
                self.prob_of_different_action = 0.0
            qsa_list = np.array([self.qsa.load(state,i) for i in range(self.num_actions)])
            qsa_std = np.std(qsa_list)
            self.qsa_std_avg = self.qsa_avg_alpha*self.qsa_std_avg + (1.0 - self.qsa_avg_alpha)*qsa_std
            noise = self.epsilon*self.qsa_std_avg*np.random.rand(self.num_actions)
            a_before = np.argmax(np.array(qsa_list))
            a = np.argmax(np.array(qsa_list + noise))
            self.prob_of_different_action = 0.999*self.prob_of_different_action + (1.0 - 0.999)*(a != a_before)

        self.tmp_a_list = np.copy(np.array(qsa_list))
        return (a,qsa_list[a])

    def choose_action_recurrence(self,state,p):
        max_action = -1e99
        
        #epsilon-greedy
        if(p['action_type'] == 'e_greedy'):
            qsa_list = []
            qsa_list_val = []
            for i in range(self.num_actions):
                v = self.qsa.load(state,i)
                qsa_list.append(np.copy(v))
                qsa_list_val.append(np.copy(self.qsa.net.layer[0].output[0:-1]))

            if(np.random.random() < self.epsilon):
                a = np.random.randint(self.num_actions)
                #print("selected action " + str(a) + "which had QSA value of: " + str(qsa_list[a]))
            else:
                a = np.argmax(np.array(qsa_list))
                #print("selected random action " + str(a) + "which had QSA value of: " + str(qsa_list[a]))
        elif(p['action_type'] == 'noisy_qsa'):
            #INIT CODE HERE
            if(self.step == 0 and self.episode == 0):
                self.qsa_std_avg = p['qsa_avg_init']
                self.qsa_avg_alpha = p['qsa_avg_alpha']
                #this will give a moving average estimate of the probability of selecting a different action
                #(used for printing only)
                self.prob_of_different_action = 0.0
            qsa_list = np.array([self.qsa.load(state,i) for i in range(self.num_actions)])
            qsa_std = np.std(qsa_list)
            self.qsa_std_avg = self.qsa_avg_alpha*self.qsa_std_avg + (1.0 - self.qsa_avg_alpha)*qsa_std
            noise = self.epsilon*self.qsa_std_avg*np.random.rand(self.num_actions)
            a_before = np.argmax(np.array(qsa_list))
            a = np.argmax(np.array(qsa_list + noise))
            self.prob_of_different_action = 0.999*self.prob_of_different_action + (1.0 - 0.999)*(a != a_before)

        self.tmp_a_list = np.copy(np.array(qsa_list))
        return (a,qsa_list[a],qsa_list_val[a])



    #this updates the internal self.results variable to reflect the latests results to be either saved or returned
    def update_results(self,p):
        self.results = {}
        #TODO: save neural network weights
        if(p['qsa_type'] == 'tabular'):
            self.results['qsa_values'] = np.array(self.qsa.data);
            self.results['state_size'] = np.array(self.state_size);
        self.results['steps_balancing_pole_list'] = np.array(self.steps_balancing_pole_list)
        self.results['steps_balancing_pole_avg_list'] = np.array(self.steps_balancing_pole_avg_list)
        self.results['state_min'] = np.array(self.state_min);
        self.results['state_max'] = np.array(self.state_max);
        self.results['num_actions'] = np.array(self.num_actions);
        self.results['epsilon'] = np.array(self.epsilon)
        self.results['epsilon_decay'] = np.array(self.epsilon_decay)
        self.results['epsilon_min'] = np.array(self.epsilon_min)
        self.results['alpha'] = np.array(self.alpha)
        self.results['gamma'] = np.array(self.gamma)
        self.results['episode'] = np.array(self.episode)
        self.results['parameters'] = p
        #TODO: save and load more hyperparameters, such as cart-pole dynamics, bounds, positive and negative reward values, and others?

    def save_results(self,filename,p):
        self.update_results(p)
        #skip saving if the parameter says not to save
        if(p.has_key('skip_saving') and p['skip_saving'] == True):
            return
        save_results(filename,self.results)

#TODO: THIS FUNCTION IS BROKEN! LOADING MAY NOT WORK PROPERLY!
#      rework this, to support neural network architecture
    def load_results(self,filename,p):
        self.results = load_h5py(filename,p)

        self.epsilon = self.results['epsilon'].value
        self.epsilon_decay = self.results['epsilon_decay'].value
        self.epsilon_min = self.results['epsilon_min'].value
        self.alpha = self.results['alpha'].value
        self.gamma = self.results['gamma'].value
        self.state_min = list(self.results['state_min'])
        self.state_max = list(self.results['state_max'])
        self.state_size = list(self.results['state_size'])
        self.episode = self.results['episode'].value
        self.num_actions = 3
        self.qsa = tabular_qsa()
        self.qsa.init(self.state_min,self.state_max,self.state_size,self.num_actions)
        self.qsa.data = np.array(self.results['qsa_values'])
        print('loaded epsilon: ' + str(self.epsilon))

    def init_sim(self,p):
        self.epsilon = p['epsilon']
        self.epsilon_decay = p.get('epsilon_decay',1.0)
        self.epsilon_min = p.get('epsilon_min',self.epsilon)
        self.gamma = p['gamma']

        ##initialize Qsa arbitrarily
        self.vel_bound = p['vel_bound']
        self.pos_bound = p['pos_bound']
        self.angle_vel_bound = p['angle_vel_bound']
        self.state_min = [0.0, -self.vel_bound, -self.pos_bound, -self.angle_vel_bound]
        self.state_max = [2*math.pi,  self.vel_bound,  self.pos_bound,  self.angle_vel_bound]

        self.episode = 0

        self.num_actions = 3
        if(p['qsa_type'] == 'tabular'):
            self.qsa = tabular_qsa()
            self.state_size = [p['angle_bins'],p['angle_vel_bins'],p['pos_bins'],p['vel_bins']]
            self.qsa.init(self.state_min,self.state_max,self.state_size,self.num_actions)
            self.alpha = p['learning_rate']
        elif(p['qsa_type'] == 'nnet'):
            self.qsa = nnet_qsa()
            self.qsa.init(self.state_min,self.state_max,self.num_actions,p)
            #The neural network has its own internal learning rate (alpha is ignored)
            self.alpha = 1.0
        elif(p['qsa_type'] == 'cartpole_nnet'):
            self.qsa = cartpole_nnet_qsa()
            self.qsa.init(self.state_min,self.state_max,self.num_actions,p)
            #The neural network has its own internal learning rate (alpha is ignored)
            self.alpha = 1.0
        elif(p['qsa_type'] == 'cluster_nnet'):
            self.qsa = cluster_nnet_qsa()
            self.qsa.init(self.state_min,self.state_max,self.num_actions,p)
            #The neural network has its own internal learning rate (alpha is ignored)
            self.alpha = 1.0
        elif(p['qsa_type'] == 'recurrent_cartpole_nnet'):

            self.state_transformer = cartpole_state_transformer()
            self.state_transformer.init(p.get('do_trig_transform',False),
                p.get('create_pomdp',False),p.get('state_dupe_count',1),p)
            print("state dupe count: " + str(p.get('state_dupe_count',1)))

            self.qsa = recurrent_cartpole_nnet_qsa()
            self.qsa.init(self.state_transformer.num_states,self.num_actions,p)
            
            #The neural network has its own internal learning rate (alpha is ignored)
            self.alpha = 1.0


if __name__ == '__main__':
    g = rl_runner_sarsa()
    p = {}
    g.run_sim(p)
