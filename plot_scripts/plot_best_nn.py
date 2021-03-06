import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cartpole.misc.save_h5py import save_results,load_results

def plot_results(results):
    res = np.array(results['steps_balancing_pole_avg_list'])
    plt.plot(np.arange(res.shape[0])+1,res)

def calc_polyfit(results):
    steps = np.array(results['steps_balancing_pole_avg_list'])
    x = np.arange(steps.shape[0])+1
    z = np.polyfit(x,steps,6)
    poly_z = np.poly1d(z)
    err = (steps - poly_z(x))
    error_std = np.sum(err**2)/20000.
    print("error_std: " + str(error_std))
    xp = np.linspace(1,20000,500)
    plt.plot(x,poly_z(x))
    return 'polyfit mse: ' + str(error_std)
    

if __name__ == '__main__':
    res_filename = '../results/clustertest/cartpole_hyperopt_nn_clustertest_GZHBARMBEOKH.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/clustertest/cartpole_hyperopt_nn_clustertest_BZEPXNSLQMIB.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/clustertest/cartpole_hyperopt_nn_clustertest_CPRWMGWHSGIZ.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    plt.legend(['Cluster-Select Best #1','Cluster-Select Best #2','Cluster-Select Best #3','Tabular Best'],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_clusterselectresults_new.png',dpi=400,bbox_inches='tight')
    plt.clf()

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_DJSIEMEONPJH.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_IURAWJGFQBOG.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_WDQKCFKGGZZO.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    plt.legend(['Rectified Linear Best #1','Rectified Linear Best #2','Rectified Linear Best #3','Tabular Best'],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_rectifiedlinearresults.png',dpi=400,bbox_inches='tight')
    plt.clf()

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_HZLRAKNGGARQ.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_YLVBYEVHZPCQ.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_PUPYPWZZCYKA.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    plt.legend(['Sigmoid Best #1','Sigmoid Best #2','Sigmoid Best #3','Tabular Best'],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_sigmoidresults.png',dpi=400,bbox_inches='tight')
    plt.clf()

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_RGBDKQHFUKTF.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_AEXIDOKOUTYV.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_BUNXRRAZMCTR.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)

    plt.legend(['Tanh Best #1','Tanh Best #2','Tanh Best #3','Tabular Best'],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_tanhresults.png',dpi=400,bbox_inches='tight')
