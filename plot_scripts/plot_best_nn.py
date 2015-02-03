import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cartpole.misc.save_h5py import save_results,load_results

def plot_results(results):
    res = np.array(results['steps_balancing_pole_avg_list'])
    plt.plot(np.arange(res.shape[0])+1,res)

if __name__ == '__main__':
#    plt.figure(1)

    res = load_results('../results/nn_initialtest/cartpole_hyperopt_nn_initialtest1_ZOYDEGQYAYVY.h5py')
    plot_results(res)

    res = load_results('../results/nn_initialtest/cartpole_hyperopt_nn_initialtest1_GIEVHKWKCOXZ.h5py')
    plot_results(res)

    res = load_results('../results/nn_initialtest/cartpole_hyperopt_nn_initialtest1_QBXTETCKLGGS.h5py')
    plot_results(res)

    res = load_results('../results/cartpole_sarsa_test_hyperopt1.1.h5py')
    plot_results(res)
    plt.legend(['NN Best #1','NN Best #2','NN Best #3','Tabular Best'],'upper left')
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_initialresults.png',dpi=400,bbox_inches='tight')
