#!/usr/bin/env python
import sys
from cartpole.sim.cartpole_sim import cartpole_sim
from cartpole.env.cartpole_environment import cartpole_environment
from cartpole.runner.runner_game import runner_game
from cartpole.runner.rl_runner_sarsa import rl_runner_sarsa
from cartpole.misc.autoconvert import autoconvert

import math
#from pygame.locals import *

def main(argv=[]):
    #Get the parameters file from the command line
    #use mnist_train__forget_params.py by default (no argument given)
    if(len(argv) > 1):
        params_file = argv[1]
    else:
        print("No Parameter File Specified")
        print("Usage: ./run_cartpole.py <Parameter_File> <parameter_1=value> ... <parameter_n=value>")
        print("Running with default parameter file: cartpole_default_params.py")
        params_file = 'cartpole_default_params.py'

    p = {}
    execfile(params_file,p)

    #grab extra parameters from command line
    for i in range(2,len(sys.argv)):
        (k,v) = sys.argv[i].split('=')
        v = autoconvert(v)
        p[k] = v
        print(str(k) + ":" + str(v))

    #the runtype tells exactly what type of simulation we want to run
    if(p['runtype'].lower() == "game"):
        run = runner_game()
        run.run_sim(p)

    if(p['runtype'].lower() == "sarsa"):
        run = rl_runner_sarsa()
        run.run_sim(p)

if __name__ == '__main__':
    main(sys.argv)
