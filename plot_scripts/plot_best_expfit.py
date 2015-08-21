import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cartpole.misc.save_h5py import save_results,load_results
from scipy.optimize import curve_fit

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

def calc_expon_fit(results,guess=None):
    steps = np.array(results['steps_balancing_pole_avg_list'])
    x = np.arange(steps.shape[0])+1

    #x = x/1000.
    #steps=steps/1000.
    #if(guess is not None):
    #    popt,pcov = curve_fit(expon_decay,x,steps,guess)
    #else:
    popt,pcov = curve_fit(expon_decay,x,steps,(max(steps),0.001,max(steps)),maxfev=2000)
    print(max(steps))
    print('optimal: ' + str(popt))
    y_exp = expon_decay(x,*popt)
    err = (steps - y_exp)
    error_std = np.sum(err**2)/20000.
    print("error_std: " + str(error_std))
    xp = np.linspace(1,20000,500)
    plt.plot(x,y_exp)
    return 'exponential fit mse: ' + str(error_std)

def expon_decay(x,a,b,c):
    b = b
    #c = c/10
    #a = a/10
    return c-a*np.exp(-b*x)

if __name__ == '__main__':
#    plt.figure(1)

#    res_filename =('../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_HDLUUALMKGZT.h5py')
#    res_filename =('../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_IURAWJGFQBOG.h5py')
#    plot_results(res)

#    res_filename =('../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_UVMIRBOZGUQN.h5py')
#    res_filename =('../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_WIUJDORHBMXE.h5py')
#    plot_results(res)

#    res_filename =('../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_PXUBSKTZOUYA.h5py')
#    res_filename =('../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_USYZBZAPOHWX.h5py')
#    plot_results(res)

    res_filename = '../results/clustertest/cartpole_hyperopt_nn_clustertest_GZHBARMBEOKH.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err1 = calc_expon_fit(res)

    res_filename = '../results/clustertest/cartpole_hyperopt_nn_clustertest_EHWITHFAYCEL.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err2 = calc_expon_fit(res)

    res_filename = '../results/clustertest/cartpole_hyperopt_nn_clustertest_BZEPXNSLQMIB.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err3 = calc_expon_fit(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err4 = calc_expon_fit(res)

    plt.legend(['Cluster-Select Best #1',p_err1,'Cluster-Select Best #2',p_err2,'Cluster-Select Best #3',p_err3,'Tabular Best',p_err4],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_clusterselectresults_new_expfit.png',dpi=400,bbox_inches='tight')
    plt.clf()

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_DJSIEMEONPJH.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err1 = calc_expon_fit(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_IURAWJGFQBOG.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err2 = calc_expon_fit(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_WDQKCFKGGZZO.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err3 = calc_expon_fit(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err4 = calc_expon_fit(res)

    plt.legend(['Rectified Linear Best #1',p_err1,'Rectified Linear Best #2',p_err2,'Rectified Linear Best #3',p_err3,'Tabular Best',p_err4],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_rectifiedlinearresults_expfit.png',dpi=400,bbox_inches='tight')
    plt.clf()

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_HZLRAKNGGARQ.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err1 = calc_expon_fit(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_YLVBYEVHZPCQ.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err2 = calc_expon_fit(res,(3.23060282e+02,1.20783727e-04,2.67788115e+02))

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_PUPYPWZZCYKA.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err3 = calc_expon_fit(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err4 = calc_expon_fit(res)

    plt.legend(['Sigmoid Best #1',p_err1,'Sigmoid Best #2',p_err2,'Sigmoid Best #3',p_err3,'Tabular Best',p_err4],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_sigmoidresults_expfit.png',dpi=400,bbox_inches='tight')
    plt.clf()

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_RGBDKQHFUKTF.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err1 = calc_expon_fit(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_AEXIDOKOUTYV.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err2 = calc_expon_fit(res)

    res_filename = '../results/nn_initialtest3/cartpole_hyperopt_nn_initialtest3_BUNXRRAZMCTR.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err3 = calc_expon_fit(res)

    res_filename = '../results/cartpole_sarsa_test_hyperopt1.1.h5py'
    print('loading: ' + res_filename)
    res = load_results(res_filename)
    plot_results(res)
    p_err4 = calc_expon_fit(res)

    plt.legend(['Tanh Best #1',p_err1,'Tanh Best #2',p_err2,'Tanh Best #3',p_err3,'Tabular Best',p_err4],'upper left',prop={'size': 6})
    plt.axis([0,20000,0,1000])

    plt.xlabel('Episode')
    plt.ylabel('Number of Steps Spent "balancing" Pole')
    plt.grid()
    plt.savefig('../result_images/cartpole_nn_tanhresults_expfit.png',dpi=400,bbox_inches='tight')
