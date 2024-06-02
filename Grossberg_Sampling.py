import numpy as np

def Grossberg_Sampling(stack, n_samples=256):
    debug = False

    _, col, stack_size = stack.shape

    if n_samples < 1:
        n_samples = 256

    # Computando el CDF
    for i in range(stack_size):
        for j in range(col):
            h_cdf = np.cumsum(stack[:, j, i])
            stack[:, j, i] = h_cdf / np.max(h_cdf)
        
        if debug:
            import matplotlib.pyplot as plt
            plt.figure(4)
            plt.plot(np.arange(256) / 255.0, stack[:, 0, i])
            plt.show()

    delta = 1.0 / (n_samples - 1)
    u = np.arange(0.0, 1.0 + delta, delta)

    stack_out = np.zeros((len(u), stack_size, col), dtype=int)

    for i in range(len(u)):
        for j in range(col):
            for k in range(stack_size):
                val = np.argmin(np.abs(stack[:, j, k] - u[i]))
                stack_out[i, k, j] = val

    return stack_out
