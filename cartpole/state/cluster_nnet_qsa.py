#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np
import cartpole.misc.cluster_select_func as csf

class cluster_nnet_qsa(object):
    def init(self,mins,maxs,num_actions,p):
        layers = [];
        self.state_size = len(list(mins))
        self.num_actions = num_actions
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.incorrect_target = p['incorrect_target']
        print(str(self.state_size) + " " + str(self.num_actions))
        self.correct_target = p['correct_target']
        layers.append(nnet.layer(self.state_size + self.num_actions))
        layers.append(nnet.layer(p['num_hidden'],p['activation_function'],
                                 initialization_scheme=p['initialization_scheme'],
                                 initialization_constant=p['initialization_constant'],
                                 dropout=p['dropout'],use_float32=p['use_float32'],
                                 momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))
        layers.append(nnet.layer(1,
                                 initialization_scheme=p['initialization_scheme_final'],
                                 initialization_constant=p['initialization_constant_final'],
                                 use_float32=p['use_float32'],
                                 momentum=p['momentum'],step_size=p['learning_rate']))

        self.net = nnet.net(layers)

        self.do_neuron_clustering=False #by default
        if(p.has_key('cluster_func') and p['cluster_func'] is not None):
            #TODO: Make sure the centroids cover the input space appropriately
            self.net.layer[0].centroids = np.asarray(((np.random.random((self.net.layer[0].weights.shape)) - 0.5) * 2.25),np.float32)
            #make the centroid bias input match the bias data of 1.0
            self.net.layer[0].centroids[:,-1] = 1.0
            #print(str(self.net.layer[0].centroids.shape))
            #print(str(self.net.layer[0].centroids))
            self.net.layer[0].select_func = csf.select_names[p['cluster_func']]
            #print('cluster_func: ' + str(csf.select_names[p['cluster_func']]))
            self.net.layer[0].centroid_speed = p['cluster_speed']
            self.net.layer[0].num_selected = p['clusters_selected']
            self.do_neuron_clustering=True #set a flag to log neurons that were used for clustering
            if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
                self.net.layer[0].do_cosinedistance = True
                print('cosine set to true')

    def store(self,state,action,value):
        s = (np.array(state) - self.mins)/(self.maxs - self.mins)
        s = s-0.5
        s = s*2.25
        action_list = np.ones((1,self.num_actions))*self.incorrect_target
        action_list[0,action] = self.correct_target
        s = np.append(s,action_list)[:,np.newaxis]
        self.net.input = s
        self.net.feed_forward()
        #print("This should be 0: " + str(self.net.output - value_tmp) + " netoutput - value: " + str(self.net.output - value))
        self.net.error = self.net.output - value
        self.net.back_propagate()
        self.net.update_weights()

    def load(self,state,action):
        s = (np.array(state) - self.mins)/(self.maxs - self.mins)
        s = s-0.5
        s = s*2.25
        action_list = np.ones((1,self.num_actions))*self.incorrect_target
        action_list[0,action] = self.correct_target
        s = np.append(s,action_list)[:,np.newaxis]
        self.net.input = s
        self.net.feed_forward()
        return self.net.output[0,0]

if __name__ == '__main__':
    #TODO: tests?
    pass
