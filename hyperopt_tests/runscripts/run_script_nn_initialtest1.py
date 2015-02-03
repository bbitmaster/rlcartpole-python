#!/usr/bin/env python
import sys
import math
import time
from hyperopt import fmin, tpe, hp, mix, rand, STATUS_OK, STATUS_FAIL
from functools import partial
import hyperopt
from hyperopt.mongoexp import MongoTrials
import launch
import evaluate_forget
import hyperopt_tests.support.launch_cartpole


space = {
    'activation_function' : hp.choice('a_function',['tanh','sigmoid','linear_rectifier']),
    'num_hidden'          : hp.choice('n_hidden',[8,16,32,64,128]),
    'learning_rate'       : hp.choice('l_rate',[0.1,0.01,0.001]),
    'learning_rate_decay' : hp.choice('l_rate_decay',[0.999,0.9992,0.9995,0.9997,0.9999,1.0]),
    'learning_rate_min'   : hp.choice('l_rate_min',[0.00001,0.0001,0.001]),
    'positive_reward'     : hp.choice('p_reward',[0.1,0.05,0.01]),
    'epsilon'             : hp.choice('eps',[0.5,1.0,1.5,2.0,5.0,10.0]),
    'epsilon_min'         : hp.choice('eps_min',[0.01,0.5,0.1,0.001]),
    'gamma'               : hp.choice('gamm',[0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.999])
    }

def objective(space):
    #This function will be unpickled by hyperopt and we need to reimport everythinng for it to work
    import time
    from hyperopt import STATUS_OK, STATUS_FAIL
    import random
    import sys
    import launch
    import evaluate_forget
    import math
    import os
    from hyperopt_tests.support.launch_cartpole import launch_cartpole
    import hyperopt_tests

    os.path.dirname(hyperopt_tests.__file__)

    #we cheat to get the directory of the parameters file. we find the directory of the hyperopt_tests import
    cwd = os.path.dirname(hyperopt_tests.__file__)
    
    paramsfile_relative = '../params/cartpole_nn_hyperopt_params.py'
    paramsfile = os.path.abspath(os.path.join(cwd,paramsfile_relative))

    params = space

    #since hyperopt is ran from somewhere else, set the results directory correctly for saving results
    params['results_dir'] = os.path.abspath(os.path.join(cwd,'../results/')) + '/'

    #set any parameteers here to override the parameters in the params file
    params['simname'] = 'cartpole_hyperopt_nn_initialtest1'

    #Give each run a random unique identifier for the version. This allows us to locate any saved results via a signature in the filename
    rnd_str = '_'
    for i in range(12):
        rnd_str += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    params['version'] = rnd_str
    t = time.time()
    
    #TODO? Maybe catch exceptions and set status to failed if the occur?
    #try:
    (obj,argmin) = launch_cartpole(paramsfile,params)

    print('obj: ' + str(obj) + ' argmin: ' + str(argmin))
    return {
        'loss' : obj,
        'argmin' : str(argmin),
        'status' : STATUS_OK,
        'eval_time': str(time.time() - t),
        'rnd_str' : rnd_str
        #any additional logging goes here
#        'attachments':
#            {'stdout' : str(stdout),
#             'stderr' : str(stderr),
#             'params' : str(params)}
        }
from hyperopt_tests.mongodb_machine import mongodb_machine
trials = MongoTrials(mongodb_machine + 'cartpole_nn_tests/jobs',exp_key='cartpole_hyperopt_nn_initialtest1')

best = fmin(objective, space, trials=trials, max_evals=1000,algo=rand.suggest)

print('best: ' + str(best))
