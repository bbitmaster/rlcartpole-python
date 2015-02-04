#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np
from math import sin, cos
from cartpole.state.nnet_qsa import nnet_qsa
#this class transforms the cartpole state variables to be a bit more meaningful before passing them
#on to the nnet_qsa class
class cartpole_nnet_qsa(nnet_qsa):
    def init(self,mins,maxs,num_actions,p):
        #first state variable (theta) is transformed to sin(theta) and cos(theta)
        mins = np.append(np.array([-1.0,-1.0]),mins[1:])
        maxs = np.append(np.array([1.0,1.0]),maxs[1:])
        nnet_qsa.init(self,mins,maxs,num_actions,p)

    def store(self,state,action,value):
        s = np.append(np.array([sin(state[0]),cos(state[0])]),state[1:])
        nnet_qsa.store(self,s,action,value)

    def load(self,state,action):
        s = np.append(np.array([sin(state[0]),cos(state[0])]),state[1:])
        return nnet_qsa.load(self,s,action)

if __name__ == '__main__':
    pass
