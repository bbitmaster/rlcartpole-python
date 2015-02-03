#!/bin/bash
set -x verbose
ssh hydra1.eecs.utk.edu -t 'tmux new -d "cd ~/research/python/tmp;OMP_NUM_THREADS=4 hyperopt-mongo-worker --mongo=hyperopt@com1577.eecs.utk.edu/cartpole_nn_tests --poll-interval=30"'
