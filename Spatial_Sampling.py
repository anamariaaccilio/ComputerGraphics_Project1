import numpy as np
from math import ceil

import Normalizar_Img
import Check_Monotonicity

def Spatial_Sampling(stack, sort_index, n_samples, sample_type='RegularSpatial'):
    r, c, col, stack_size = stack.shape

    min_samples = max(round(r * c * 0.001), 512)
    
    if n_samples < 1:
        n_samples = min_samples
    
    X = []
    Y = []

    if sample_type == 'RandomSpatial':
        X = Normalizar_Img(np.round(np.random.rand(n_samples) * (c - 1)).astype(int), 0, c - 1)
        Y = Normalizar_Img(np.round(np.random.rand(n_samples) * (r - 1)).astype(int), 0, r - 1)
    elif sample_type == 'RegularSpatial':
        f = round(np.sqrt(n_samples) + 1)
        rate_x = max(ceil(c / f), 1)
        rate_y = max(ceil(r / f), 1)

        X, Y = np.meshgrid(range(0, c, rate_x), range(0, r, rate_y))
        X = X.flatten()
        Y = Y.flatten()
        n_samples = len(X)

    stack_out = np.zeros((n_samples, stack_size, col), dtype=stack.dtype)

    c = 0
    for i in range(n_samples):
        tmp = np.zeros((stack_size, col), dtype=stack.dtype)
        for j in range(col):
            for k in range(stack_size):
                tmp[k, j] = stack[Y[i], X[i], j, k]
        
        if (Check_Monotonicity(sort_index, tmp)):
            stack_out[c, :, :] = tmp
            c += 1

    stack_out = stack_out[:c, :, :]

    return stack_out
