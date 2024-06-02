import numpy as np
from scipy.linalg import lstsq

def Gsolve(Z, B, l, w):
    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros(A.shape[0])
    
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k] = wij * B[j]
            k += 1
    
    A[k, 128] = 1
    k += 1
    
    for i in range(1, n - 1):
        A[k, i - 1] = l * w[i]
        A[k, i] = -2 * l * w[i]
        A[k, i + 1] = l * w[i]
        k += 1
    
    x = lstsq(A, b)[0]
    g = x[:n]
    
    return g
