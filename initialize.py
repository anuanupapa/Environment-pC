import numpy as np
import numba as nb

@nb.njit
def init_arr(N, p_init = 0.65):
    action = np.zeros((N, 1))
    
    for ind in range(N):
        action[ind] = nb.float64(np.random.random() < p_init)
    return action

@nb.njit
def init_wealth(N, g):
    money = np.zeros((N, 1))
    if g == 0.2:
        for i in range(N):
            money[i] = 300. + 400. * nb.float64(np.random.random()<0.5)
    elif g == 0.4:
        for i in range(N):
            money[i] = 250. + 900. * nb.float64(np.random.random()<0.5)
    else:
        for i in range(N):
            money[i] = 500.
    return money
        
@nb.njit
def init_adjmat(N, p):
    adjacency_matrix = np.zeros((N, N))
    for i in range(N):
        for j in np.arange(i+1, N, 1):
            response = nb.float64(np.random.random() < p)
            adjacency_matrix[i][j] = response
            adjacency_matrix[j][i] = response
    return(adjacency_matrix)


if __name__ == "__main__":
    print(init_arr(5, 1., 1.))
    print(init_adjmat(5, 0.3))


