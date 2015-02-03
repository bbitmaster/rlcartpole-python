#This is just a test script I created for testing and displaying sample spaces
from hyperopt import hp

space = {
    'activation_function' : hp.choice('a_function',['tanh','sigmoid','linear_rectifier']),
    'num_hidden' : hp.choice('n_hidden',[8,16,32,64,128]),
    'learning_rate' : hp.choice('l_rate',[0.1,0.01,0.001]),
    'learning_rate_decay' : hp.choice('l_rate_decay',[0.999,0.9992,0.9995,0.9997,0.9999,1.0]),
    'learning_rate_min' : hp.choice('l_rate_min',[0.00001,0.0001,0.001]),
    'positive_reward' : hp.choice('p_reward',[0.1,0.05,0.01]),
    'epsilon' : hp.choice('eps',[0.5,1.0,1.5,2.0,5.0,10.0]),
    'epsilon_min' : hp.choice('eps_min',[0.01,0.5,0.1,0.001]),
    'gamma' : hp.choice('gamm',[0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.999])
    }
import hyperopt.pyll.stochastic
print hyperopt.pyll.stochastic.sample(space)    
