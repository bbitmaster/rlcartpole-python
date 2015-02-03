#!/usr/bin/env python
import subprocess
import os
import sys
import numpy as np
from cartpole.runner.main_runner import main_runner

def launch_cartpole(paramsfile,params):

    m = main_runner()
    results = m.run(paramsfile,params)
    #compute objective from results
    obj = np.max(results['steps_balancing_pole_avg_list'])
    argmax = np.argmax(results['steps_balancing_pole_avg_list'])
    return (obj,argmax)

if __name__ == '__main__':
    p = {}
    print("launching neural network test with 100 episodes of training")
    p['train_episodes'] = 3000
    p['skip_saving'] = True
    obj = launch_cartpole('../../params/cartpole_nn_default_params.py',p)
    print("objective function was: " + str(obj))
