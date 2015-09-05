#this class implements Neural Network storage for a Qsa table
from nnet_toolkit import nnet
import numpy as np
from math import sin, cos
import cartpole.misc.cluster_select_func as csf

class recurrent_cartpole_nnet_qsa(object):
    def init(self,state_size,num_actions,p):
        layers = [];

        self.do_trig_transform = False
        if(p.get('do_trig_transform',True)):
            self.do_trig_transform = True


        self.state_size = state_size
        self.num_actions = num_actions

        self.action_dupe_count = p.get('action_dupe_count',1)

        self.do_recurrence = False
        if(p['do_recurrence']):
            input_size = self.state_size + self.num_actions*self.action_dupe_count + p['num_hidden']
            self.do_recurrence = True
        else:
            input_size = self.state_size + self.num_actions*self.action_dupe_count

        self.learning_rate = p['learning_rate']

        print("state size        : " + str(self.state_size))
        print("num actions       : " + str(self.num_actions))
        print("action dupe count : " + str(self.action_dupe_count))
        print("num hidden        : " + str(p['num_hidden']))
        print("input size        : " + str(input_size))

        self.incorrect_target = p['incorrect_target']
        print(str(self.state_size) + " " + str(self.num_actions))
        self.correct_target = p['correct_target']
        layers.append(nnet.layer(input_size))
        layers.append(nnet.layer(p['num_hidden'],p['activation_function'],
                                 initialization_scheme=p['initialization_scheme'],
                                 initialization_constant=p['initialization_constant'],
                                 dropout=p['dropout'],use_float32=p['use_float32'],
                                 momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))
        if(p.has_key('num_hidden2') and p['num_hidden2'] is not None):
            layers.append(nnet.layer(p['num_hidden2'],p['activation_function2'],
                                 initialization_scheme=p['initialization_scheme2'],
                                 initialization_constant=p['initialization_constant2'],
                                 dropout=p['dropout2'],use_float32=p['use_float32'],
                                 momentum=p['momentum2'],maxnorm=p['maxnorm2'],step_size=p['learning_rate2']))
 

        layers.append(nnet.layer(1,
                                 initialization_scheme=p['initialization_scheme_final'],
                                 initialization_constant=p['initialization_constant_final'],
                                 use_float32=p['use_float32'],
                                 momentum=p['momentum'],step_size=p['learning_rate']))

        self.net = nnet.net(layers)

        if(p.has_key('cluster_func') and p['cluster_func'] is not None):
            print("layer 0 has cluster func")
            self.cluster_func = p['cluster_func']
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
            if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
                self.net.layer[0].do_cosinedistance = True
                print('cosine set to true')
            if(p.has_key('zeta_decay') and p['zeta_decay'] is not None):
                self.net.layer[0].zeta_matrix = np.ones(self.net.layer[0].weights.shape,dtype=np.float32)
                self.net.layer[0].zeta = 1.0
                self.zeta_decay = p['zeta_decay']


        if(p.has_key('cluster_func2') and p['cluster_func2'] is not None):
            print("layer 1 has cluster func")
            self.cluster_func2 = p['cluster_func2']
            #TODO: Make sure the centroids cover the input space appropriately
            self.net.layer[1].centroids = np.asarray(((np.random.random((self.net.layer[1].weights.shape)) - 0.5) * 2.25),np.float32)
            #make the centroid bias input match the bias data of 1.0
            self.net.layer[1].centroids[:,-1] = 1.0
            #print(str(self.net.layer[0].centroids.shape))
            #print(str(self.net.layer[0].centroids))
            self.net.layer[1].select_func = csf.select_names[p['cluster_func2']]
            #print('cluster_func: ' + str(csf.select_names[p['cluster_func']]))
            self.net.layer[1].centroid_speed = p['cluster_speed2']
            self.net.layer[1].num_selected = p['clusters_selected2']
            if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
                self.net.layer[1].do_cosinedistance = True
                print('cosine set to true')
            if(p.has_key('zeta_decay2') and p['zeta_decay2'] is not None):
                self.net.layer[1].zeta_matrix = np.ones(self.net.layer[1].weights.shape,dtype=np.float32)
                self.net.layer[1].zeta = 1.0
                self.zeta_decay = p['zeta_decay2']



        if(p.has_key('_lambda') and p['_lambda'] is not None):
            self._lambda = p['_lambda']
            self.gamma = p['gamma']
            for l in self.net.layer:
                l.eligibility = l.gradient


        print("Network Size Check:" )
        print("  weight 0 size: " + str(self.net.layer[0].weights.shape))
        if(len(self.net.layer) > 1):
            print("  weight 1 size: " + str(self.net.layer[1].weights.shape))
        if(len(self.net.layer) > 2):
            print("  weight 2 size: " + str(self.net.layer[2].weights.shape))
        if(len(self.net.layer) > 3):
            print("  weight 3 size: " + str(self.net.layer[3].weights.shape))


    def store(self,state,action,value):
        action_list = np.ones((self.num_actions),dtype=np.float32)*self.incorrect_target
        action_list[action] = self.correct_target
        action_list = np.tile(action_list,self.action_dupe_count)

        s = np.append(state,action_list)
        #print("state: " + str(s[0:3]))
        self.net.input = s[:,np.newaxis]
        self.net.feed_forward()
        #print("This should be 0: " + str(self.net.output - value_tmp) + " netoutput - value: " + str(self.net.output - value))

        #eligibility update
        if(hasattr(self,'_lambda')):
            self.net.error = np.ones((1,1),dtype=np.float32)
            self.net.back_propagate()
            for l in self.net.layer:
                l.eligibility = self.gamma*self._lambda*l.eligibility + l.gradient
                delta_t = -(value - self.net.output)
                l.gradient = l.eligibility*delta_t
        else:
            self.net.error = self.net.output - value
            self.net.back_propagate()

        if(hasattr(self.net.layer[0],'zeta')):
        #decay zeta
            self.net.layer[0].zeta = self.net.layer[0].zeta*self.zeta_decay
            self.net.layer[0].zeta_matrix[:] = self.net.layer[0].zeta
            self.net.layer[0].step_size = (1.0 - self.net.layer[0].zeta_matrix)*self.learning_rate

            self.net.update_weights()
            #move centroids
            csf.update_names[self.cluster_func](self.net.layer[0])
        elif(hasattr(self.net.layer[1],'zeta')):
        #decay zeta
            self.net.layer[1].zeta = self.net.layer[1].zeta*self.zeta_decay
            self.net.layer[1].zeta_matrix[:] = self.net.layer[1].zeta
            self.net.layer[1].step_size = (1.0 - self.net.layer[1].zeta_matrix)*self.learning_rate

            self.net.update_weights()
            #move centroids
            csf.update_names[self.cluster_func2](self.net.layer[1])
        else:
            self.net.update_weights()

    def load(self,state,action):
        action_list = np.ones((self.num_actions),dtype=np.float32)*self.incorrect_target
        action_list[action] = self.correct_target
        action_list = np.tile(action_list,self.action_dupe_count)
        s = np.append(state,action_list)
        
        self.net.input = s[:,np.newaxis]
        self.net.feed_forward()
        return self.net.output[0,0]

if __name__ == '__main__':
    #TODO: tests?
    pass
