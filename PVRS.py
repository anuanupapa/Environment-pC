import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as scsp
import networkx as nx
import time
import numba as nb
from numba import int64, float64
import seaborn as sns

import PGG
import Record
import rewire
import initialize as init

sns.set()

tTime = time.time()
totTime = 0.

totNP = 50
trials = 100
rounds = 20

c = 50.
r = 2.
lam = 0.001
p0 = 0.7
b = 0.1
ini_re = 0.3

coopfrac_arr = np.zeros((trials, rounds))
deg_arr = np.zeros((trials, rounds))
gini_arr = np.zeros((trials, rounds))

for it in range(trials):
    
    AdjMat = init.init_adjmat(totNP, ini_re)
    actA = init.init_arr(totNP)
    cpayA = init.init_wealth(totNP, 0.2)
    
    for i_main in range(rounds):
        
        # Play PGG
        pay = PGG.game(AdjMat, actA, r, c, totNP)

        # Record features
        [coopfrac, deg, gini] = Record.measure(actA, cpayA, AdjMat)
        coopfrac_arr[it, i_main] = coopfrac
        deg_arr[it, i_main] = deg
        gini_arr[it, i_main] = gini
        
        # Note rewire is being done before updating wealth
        # As rewiring does not require wealth
        
        # Rewire
        AdjMat = rewire.rewiring_process(AdjMat, actA, 0.3)

        # Decide
        [actA, cpayA] = Record.update_iterated(pay, AdjMat, cpayA,
                                               actA, p0, b, lam, totNP)

print(time.time()-tTime)

trounds = np.arange(0, rounds, 1)
coopfrac_arr_avg = np.mean(coopfrac_arr, axis = 0)
coopfrac_arr_std = np.std(coopfrac_arr, axis = 0)

deg_arr_avg = np.mean(deg_arr, axis = 0)
deg_arr_std = np.std(deg_arr, axis = 0)

gini_arr_avg = np.mean(gini_arr, axis = 0)
gini_arr_std = np.std(gini_arr, axis = 0)


plt.plot(trounds, coopfrac_arr_avg, 'o-', color='b')
plt.fill_between(trounds, coopfrac_arr_avg - coopfrac_arr_std, coopfrac_arr_avg + coopfrac_arr_std, alpha=0.5, color='b')
plt.ylim(0, 1)
plt.show()

plt.plot(trounds, deg_arr_avg, 'o-', color='b')
plt.fill_between(trounds, deg_arr_avg - deg_arr_std, deg_arr_avg + deg_arr_std, alpha=0.5, color='b')
plt.ylim(0, 35)
plt.show()

plt.plot(trounds, gini_arr_avg, 'o-', color='b')
plt.fill_between(trounds, gini_arr_avg - gini_arr_std, gini_arr_avg + gini_arr_std, alpha=0.5, color='b')
plt.ylim(0, 0.5)
plt.show()
# -------------------------------------------------
