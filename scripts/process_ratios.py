import sys
import numpy as np
import pandas as pd
from math import ceil, floor


dsets = {
    "shakespeare": [15, 25, 35, 50, 75],
    "femnist": [8, 16, 24, 32, 40],
    "celeba": [4, 8, 11, 15, 19],
    "synthetic": [4, 9, 13, 18, 22],
    "sent140": [3, 7, 10, 13, 17]
}

if len(sys.argv) != 2:                                                          
    print("Usage: python3 " + sys.argv[0] + " [DATASET]")
    sys.exit()

if sys.argv[1] not in dsets:
    ds = dsets.keys()
    print("[DATASET] must be in {" + ", ".join(ds) + "}")
    sys.exit()

dataset = sys.argv[1] 
df = pd.read_csv('leaf_experiments.csv')
df.columns=['dataset','i','u','g','acc1','acc2','niter']
df = df.loc[lambda dframe: dframe['dataset'] == dataset, :]\
                [['i','u','g','niter']]

us = dsets[dataset]
ps = [1.25, 1, 0.75, 0.5, 0.25]

def rround(g_unrounded):
    # In some python versions, round(2.5) = 2
    # Below code fixes it
    g_l = floor(g_unrounded)
    g_h = ceil(g_unrounded)
    g_m = (g_l+g_h)/2
    if(g_unrounded >= g_m):
        return g_h
    else:
        return g_l 

done=set()
for u in us:
    framesG=[]
    framesNG=[]
    for p in ps:
        g = rround(u*p)
        ug_tuple=(u,g)
        if ug_tuple not in done:
            print(str(u) + " " + str(g))
            ug = df[(df['g'] == g) & (df['u'] == u)]
            grouped = ug.groupby(['u','g'], as_index=False)\
                        .agg(stepswguess=('niter', np.mean), \
                             std1=('niter', np.std), \
                             count1=('niter', 'count'))
            framesG.append(grouped)
            noguess = df[(df['g'] == 0) & (df['u'] == u+g)]\
                            .groupby(['u','g'], as_index=False)\
                            .agg(stepswoguess=('niter', np.mean), \
                                 std2=('niter', np.std), \
                                 count2=('niter', 'count'))
            framesNG.append(noguess)
            done.add(ug_tuple)
    wg = pd.concat(framesG)
    wg['ug'] = wg['u'] + wg['g']
    wog = pd.concat(framesNG).rename(columns={'u':'ug'})
    m = wg.merge(wog, how='inner', on='ug')
    final = m[['u','g_x','stepswguess', 'stepswoguess', \
               'std1', 'std2', 'count1', 'count2']].rename(columns={'g_x':'g'})
    final.to_csv(dataset + '-u' + str(u) + '.csv', index=False)
