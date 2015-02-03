#!/usr/bin/env python
import subprocess
import os
import h5py as h5
import numpy as np

def evaluate_forget(cwd,paramsfile,params):
    paramsfile_dict = {}
    execfile(os.path.join(cwd,paramsfile),paramsfile_dict)

    p = dict(paramsfile_dict.items() + params.items())
    fname = os.path.join(cwd,p['results_dir'] + p['simname'] + p['version'] + '.h5py')
    print('DBG: opening: ' + fname)
    try:
        f_handle = h5.File(fname,'r')
    except IOError:
        return ('fail',1000,1000)
    m1 = np.array(f_handle['test_missed_percent1_list'])
    m2 = np.array(f_handle['test_missed_percent2_list'])
    f_handle.close()
    print('DBG: m1 and m2 shapes: ' + str(m1.shape) + ' ' + str(m2.shape))
    m = m1+m2
    obj = np.min(m)
    argmin = np.argmin(m)
    return ('ok',obj,argmin)

if __name__ == '__main__':
    p = {}
    p['simname'] = 'nonstationary_initialclusteringsubsettest1234567890_nopca_midlearningrate'
    p['version'] = 'v1.4'
    p['results_dir'] = '../results/'
    
    obj = evaluate_forget('/home/bgoodric/research/python/nn_experiments/nn_tests/','mnist_train_nonstationary_cluster_subset_params.py',p)
    print('objective: ' + str(obj))
