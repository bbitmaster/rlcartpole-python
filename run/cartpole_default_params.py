#cartpole_default_params
runtype='game'

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'cartpole_sarsa_test'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

save_interval = 30*60

use_float32=True

random_seed = 4;


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

epsilon=0.10
gamma=0.8
alpha=0.4

#If defined, will print the state variables on every frame
print_state_debug=True

#in sarsa mode, this tells if the SDL display should be enabled. Set to False if the machine does not have pygame installed
do_vis=True

#in sarsa mode, this tells how often to display, -1 for none
showevery=300

#these affect the display. They tell the size in pixels of the display, the axis size, and how many frames to skip
display_width=1024
display_height=768
axis_x_min=-20
axis_x_max=20
axis_y_min=-20
axis_y_max=20
fps=60
