#!/usr/bin/env python
import subprocess
import os
import h5py as h5
import numpy as np
import sys
from operator import itemgetter
from cartpole.misc.save_h5py import save_results,load_results

def evaluate_params(fname):
    results = load_results(fname)
    results['obj'] = np.max(np.array(results['steps_balancing_pole_avg_list']))
    results['argmax'] = np.argmax(np.array(results['steps_balancing_pole_avg_list']))
    return results

def print_sorted(p_list):
    new_p = reversed(sorted(p_list,key=itemgetter('obj'),reverse=True))

    print("")
    for p in new_p:
        #print(str(p))
        print("Filename: " + str(p['f_name']))
        print("obj: " + str(p['obj']))
        print("argmax: " + str(p['argmax']))
        print("Parameters: ")
        for k,v in p['parameters'].items():
            #print(str(param))
            #for k,v in param.items():
            print("\t" + str(k) + " : " + str(v))

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("usage: evaluate <path>")
        sys.exit()
    path=sys.argv[1]
    p_list = []
    print("path:" + str(path))
    for root, dirs, files in os.walk(path):
        for f in files:
            path = os.path.join(root,f)
            #print('loading... ' + path)
            sys.stdout.write('.')
            sys.stdout.flush()
            p = evaluate_params(path)
            p['f_name'] = path
            p_list.append(p)
    print_sorted(p_list)

