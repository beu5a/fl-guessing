import subprocess
from math import floor
import time

dataset = 'synthetic'
batch_size = 5

# Caches computed (u, g) combinations
ts = [(3, 2), (5, 0), (3, 3)]

# List of u
us = [3, 7, 11, 15, 19]

# List of g in percentage
gs = [0.25, 0.5, 0.75, 1.0, 1.25]

# Total stochastic runs
runs = 3
tn = len(us) * len(gs) * 2
n = 1

def run_main(u, g, r):
    start_time = time.time()
    log_dir = '../logs/guessing_limits/' + dataset + '/' + str(u) + '_' + str(g) + '/' + 'r' + str(r+1)
    arguments = [
        'python', 
        'main.py',
        '-d', dataset,
        '-traindir', '../leaf/data/' + dataset + '/data/train',
        '-testdir', '../leaf/data/' + dataset + '/data/test', 
        '-r', '0.1',
        '-b', str(batch_size),
        '-m', str(u),
        '-s', '0',
        '-l', log_dir,
        '-c', 'fastest',
        '-n', '20',
        '-ee', '3',
        '-g', str(g)
        ]
    process = subprocess.run(arguments, universal_newlines=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end_time = time.time()
    print('Successful |' if(process.returncode == 0) else 'Failed |', \
        'Time taken:', '{:.3f}'.format((end_time-start_time)/60), 'mins')

for u in us:
    for gp in gs:
        g = floor(gp*u)
        print("------------ Running ", n, "/", tn)
        
        if((u, g) not in ts):
            ts.append((u, g))
            print('u =', u, 'g = ', g)
            
            for r in range(runs):            
                print('Stochastic run', r+1, '/', runs)
                run_main(u, g, r)
        
        n += 1        
        print("------------ Running ", n, "/", tn)
        
        if((u+g, 0) not in ts):
            ts.append((u+g, 0))
            print('u =', u+g, 'g = ', 0)
            
            for r in range(runs):            
                print('Stochastic run', r+1, '/', runs)
                run_main(u+g, 0, r)        
        
        n += 1
print("List of all done experiments:", ts)