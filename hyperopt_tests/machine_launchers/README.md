This directory will contain scripts to launch the hyperopt children processes on many machines.

These will consist of bash scripts with commands similar to the following
#!/bin/bash
set -x verbose
ssh <MACHINE> -t 'tmux new -d "cd ~/<BASE_DIRECTORY>;OMP_NUM_THREADS=4 hyperopt-mongo-worker --mongo=hyperopt@<MACHINE>/<JOB> --poll-interval=30"'

where <MACHINE> is the host name of the machine, <BASE_DIRECTORY> is the directory of the mongo worker, <JOB> is the mongodb database name for the job to laucnh
