#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np

class nnet_qsa(object):
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

    def store(self,state,action,value):
        s = (np.array(state) - self.mins)/(self.maxs - self.mins)
        s = s-0.5
        s = s*2.25
        action_list = np.ones((1,self.num_actions))*self.incorrect_target
        action_list[0,action] = self.correct_target
        s = np.append(s,action_list)[:,np.newaxis]
#        print(str(s))
        self.net.input = s
#        print(str(s))
        self.net.feed_forward()
        self.net.error = self.net.output - value
        self.net.back_propagate()
        self.net.update_weights()

    def load(self,state,action):
        s = (np.array(state) - self.mins)/(self.maxs - self.mins)
        s = s*1.50
        action_list = np.ones((1,self.num_actions))*self.incorrect_target
        action_list[0,action] = self.correct_target
        s = np.append(s,action_list)[:,np.newaxis]
        self.net.input = s
        self.net.feed_forward()
        return self.net.output[0,0]

if __name__ == '__main__':
    #TODO: tests?
    pass
