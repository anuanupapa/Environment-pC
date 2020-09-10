import numpy as np
import numba as nb
from numba import float64
from numba import guvectorize


@nb.njit
def measure(action, money, AM):
    coop = np.sum(action)/len(action)
    deg = np.sum(np.sum(AM, axis = 1), axis = 0)/(len(action))
    gini = gini_measure(money)
    return(coop, deg, gini)

@nb.njit
def gini_measure(mon):
    N = len(mon)
    sumval = 0.
    mu = np.mean(mon)
    for i in mon:
        for j in mon:
            sumval = sumval + np.absolute(i[0]-j[0])
    return sumval/(2*mu*N*N)



@nb.njit
def update_iterated(payoff, AM, cpay, act, p0, b, lam, N):

    # ----------------------------------------
    # PVRS procedure
    # ----------------------------------------
    actM = np.zeros_like(act)
    cpayM = update_cpay(cpay, payoff)
    
    for p in range(N):    
        neigh = find_neighs(AM, p)
        delW = np.sum(cpayM[neigh])/len(neigh) - cpayM[p]
        Cneigh = nb.float64(np.sum(act[neigh]))
        Dneigh = nb.float64(len(neigh) - Cneigh)
        pcM = update_pc_PVRS(delW, Cneigh, Dneigh, p0, b, lam)[0]
        actM[p] = update_act_PVRS(pcM)

    return actM, cpayM

@nb.vectorize
def update_pay(p):
    payMod = p
    return payMod

@nb.vectorize
def update_cpay(cp, p):
    cpayMod = cp + p
    return cpayMod

@nb.njit
def find_neighs(AM, i):
    neigh = np.where(AM[i] == 1)[0]
    return neigh

@nb.njit
def update_pc_PVRS(dW, Cn, Dn, p0, b, l):
    if Cn >= Dn:
        p = (p0 - b) + (1 - p0) * np.tanh(dW * l)
    if Cn < Dn:
        p = 1 - ((p0 - b) + (1 - p0) * np.tanh(dW * l))
    return p

@nb.njit
def update_act_PVRS(p):
    a = nb.float64(np.random.random() <=  p)
    return a



if __name__ == "__main__":
    act = np.ones(5)
    sats = np.ones(5)
    act[3]=0.
    sats[2]=0.
    am = 0.
    mea = measure(act, sats, am)
    print(mea)
    a = np.array([1,2,3,4])
    b = np.array([100,200,300,400])
    #print(update(a, b))
    print(check(a))
