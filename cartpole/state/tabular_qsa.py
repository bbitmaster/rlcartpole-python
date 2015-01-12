#this class implements a tabular storage for a Qsa table
import numpy as np

class tabular_qsa(object):
    
    def init(self,mins,maxs,size,num_actions):
        self.state_size = np.array(size)
        self.num_actions = num_actions
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        size = np.append(np.array(size),num_actions)
        self.data = -np.random.random(size)/10.0

    def store(self,state,action,value):
        s = (np.array(state) - self.mins)/(self.maxs - self.mins)
        s = s*self.state_size
        s = np.append(s,action)
        self.data[tuple(np.floor(s))] = value;

    def load(self,state,action):
        s = (np.array(state) - self.mins)/(self.maxs - self.mins)
        s = s*self.state_size
        s = np.append(s,action)
        return self.data[tuple(np.floor(s))]

if __name__ == '__main__':
    mins = [-10,-10,-10,-10]
    maxs = [10,10,10,10]
    size = [20,20,20,20]
    print("mins -10,-10,-10   maxs 10,10,10   size 20,20,20 actions: 4")
    print("initializaing discrete states with 0.0")

    qsa = tabular_qsa()
    qsa.init(mins,maxs,size,4)

    print("storing 2.0 into state 3.0,3.0,3.0,3.0 action:0")
    state = [3.0,3.0,3.0,3.0]
    qsa.store(state,0,2.0);

    state = [3.0,3.0,3.0,3.0]
    val  = qsa.load(state,0);
    print("state 3.0,3.0,3.0,3.0 action 0: " + str(val))

    val  = qsa.load(state,1);
    print("state 3.0,3.0,3.0,3.0 action 1: " + str(val))

    state = [3.6,3.0,3.0,3.0]
    val  = qsa.load(state,0);
    print("state 3.6,3.0,3.0,3.0 action 0: " + str(val))

    state = [4.0,3.0,3.0,3.0]
    val  = qsa.load(state,0);
    print("state 4.0,3.0,3.0,3.0 action 0: " + str(val))

