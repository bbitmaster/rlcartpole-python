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
save_interval = 30*60

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
qsa_type='cluster_nnet'

#parameters for neural network qsa
activation_function='tanh'

num_hidden=32
learning_rate = 0.01
learning_rate_decay_type='geometric'
learning_rate_decay=1.0
learning_rate_min=0.00001
momentum=0.0
maxnorm=None
dropout=None

initialization_scheme='glorot'
initialization_scheme_final='glorot'

initialization_constant=1.0
initialization_constant_final=1.0

#cluster_func stuff
cluster_func = None
clusters_selected=0
cluster_speed=0.0

activation_function2='linear'

num_hidden2=2048
learning_rate2 = 0.01
momentum2=0.0
maxnorm2=None
dropout2=None

initialization_scheme2='glorot'
initialization_scheme_final2='glorot'

initialization_constant2=1.0
initialization_constant_final2=1.0

earlyendepisode0=5000
earlyendreward0=30

earlyendepisode1=10000
earlyendreward1=80

#cluster_func stuff
cluster_func2 = 'cluster_func'
clusters_selected2=32
cluster_speed2=0.0


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
positive_reward = 0.01
no_reward = 0.0

#decay_type can be 'geometric' or 'linear'
decay_type='geometric'
epsilon=0.05
epsilon_min=0.005
#epsilon_decay=exp((log(epsilon_min) - log(epsilon))/10000.0)
#epsilon_decay = (epsilon - epsilon_min)/15000
epsilon_decay=0.9997
gamma=0.99

action_type='e_greedy'
#action_type='noisy_qsa'
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

