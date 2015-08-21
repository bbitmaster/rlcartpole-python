#this class implements a tabular storage for a Qsa table
import numpy as np
import math

class cartpole_state_transformer(object):
    #sets the normalization range
    def init(self,do_trig_transform,create_pomdp,state_dupe_count,p):

        self.create_pomdp = create_pomdp
        self.do_trig_transform = do_trig_transform
        self.state_dupe_count = state_dupe_count


        self.vel_bound = p['vel_bound']
        self.pos_bound = p['pos_bound']
        self.angle_vel_bound = p['angle_vel_bound']

        if(create_pomdp):
            self.state_min = np.array([0.0, -self.pos_bound])
            self.state_max = np.array([2*math.pi,  self.pos_bound])
        else:
            self.state_min = np.array([0.0, -self.vel_bound, -self.pos_bound, -self.angle_vel_bound])
            self.state_max = np.array([2*math.pi,  self.vel_bound,  self.pos_bound,  self.angle_vel_bound])

        if(do_trig_transform):
            self.state_min = np.append(np.array([-1.0,-1.0]),self.state_min[1:])
            self.state_max = np.append(np.array([1.0,1.0]),self.state_max[1:])
        if(state_dupe_count > 1):
            self.state_min = np.tile(self.state_min,state_dupe_count)
            self.state_max = np.tile(self.state_max,state_dupe_count)

        self.num_states = len(self.state_min)
            
        self.transform_class=None

    def set_transform_class(self,transform_class):
        self.transform_class = transform_class

    def transform(self,state):
        state = np.array(state)
        if(self.create_pomdp == True):
            state = np.array((state[0],state[2]))
        if(self.do_trig_transform):
            state = np.append(np.array([math.sin(state[0]),math.cos(state[0])]),state[1:])
        if(self.state_dupe_count > 1):
            state = np.tile(state,self.state_dupe_count)

        s = state.astype(np.float32)
        s = (np.array(state) - self.state_min)/(self.state_max - self.state_min)
        s = s-0.5
        s = s*2.25

        if(self.state_dupe_count > 1):
            state = np.tile(state,self.state_dupe_count)

        if(self.transform_class is None):
            return s
        else:
            return self.transform_class.transform(s)

if __name__ == '__main__':
    pass
