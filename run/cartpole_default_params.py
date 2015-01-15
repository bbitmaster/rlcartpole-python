from math import exp, log
#cartpole_default_params
runtype='game'

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
train_episodes=10000

use_float32=True

random_seed = 4;

save_images=True
image_save_dir="/home/bgoodric/tmp2/" #I Guess that underutilized windows partitition with all that storage is good for something...


#how far to bound each state variable before the simulation is considered invalid
vel_bound = 15;
angle_vel_bound = 15;
pos_bound = 10

#how many bins to use for each parameter with Discrete Qsa storage
angle_bins=20
angle_vel_bins=20
pos_bins=10
vel_bins=20

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
negative_reward = -10.0
positive_reward = 0.1
no_reward = 0.0

epsilon=0.05
epsilon_min=0.007
epsilon_decay=exp((log(epsilon_min) - log(epsilon))/10000.0)
gamma=0.99
alpha=0.4

#If defined, will print the state variables on every frame
print_state_debug=True

#in sarsa mode, this tells if the SDL display should be enabled. Set to False if the machine does not have pygame installed
do_vis=True

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

