from math import exp, log
#cartpole_default_params
runtype='sarsa'

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'cartpole_sarsa_test'
version = '1.1'
results_dir = '../results/'

#if load_name is set, then the simulation will load this file and resume from there, this is useful for watching the behavior of a trained agent
#load_name = '../results/cartpole_sarsa_test1.1.h5py'

data_dir = '../data/'
save_interval = 1*60

#run for a total number of episodes
train_episodes=20000
max_steps=1000

use_float32=True

random_seed = 4;

save_images=False
image_save_dir="/home/bgoodric/tmp2/" #I Guess that underutilized windows partitition with all that storage is good for something...


#how far to bound each state variable before the simulation is considered invalid
vel_bound = 15;
angle_vel_bound = 15;
pos_bound = 10

#storage types are 'tabular' and 'nnet'
qsa_type='nnet'

#parameters for neural network qsa
activation_function='tanh'

num_hidden=64
learning_rate = 0.004
learning_rate_decay_type='geometric'
learning_rate_decay=0.9999
learning_rate_min=0.0001
momentum=0.0
maxnorm=None
dropout=None

initialization_scheme='glorot'
initialization_scheme_final='glorot'

initialization_constant=1.0
initialization_constant_final=1.0

#action is encoded using one hot encoding with these as the "hot" and "not hot" targets
incorrect_target = -1.0
correct_target = 1.0

#cart-pole hyperparemeters
g=9.81  #gravity
l=0.5   #pole length
mp=0.01 #pole mass
mc=1.0  #cart mass

#delta-t, how much time to pass between frame steps
dt=0.02

#how hard to push left and right
push_force = 10.0

#reinforcement learning parameters
negative_reward = -1.0
positive_reward = 0.1
no_reward = 0.0

#decay_type can be 'geometric' or 'linear'
decay_type='linear'
epsilon=2.0
epsilon_min=0.1
#epsilon_decay=exp((log(epsilon_min) - log(epsilon))/10000.0)
epsilon_decay = (epsilon - epsilon_min)/15000
gamma=0.99

#action_type='e_greedy'
action_type='noisy_qsa'
qsa_avg_alpha = 0.99
qsa_avg_init  = 0.0

#If defined, will print the state variables on every frame
print_state_debug=True

#in sarsa mode, this tells if the SDL display should be enabled. Set to False if the machine does not have pygame installed
do_vis=False

#in sarsa mode, this tells how often to display, -1 for none
showevery=500

#these affect the display. They tell the size in pixels of the display, the axis size, and how many frames to skip
display_width=1280
display_height=720
axis_x_min=-10.0
axis_x_max=10.0
axis_y_min=-5.5
axis_y_max=5.5
fps=60

