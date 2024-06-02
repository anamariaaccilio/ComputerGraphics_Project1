import numpy as np
from scipy.optimize import fmin

def Find_Chromaticy_Scale(M, I):
    
    if len(M) != len(I) or len(M) == 0 or len(I) == 0:
        raise ValueError('Find_Chromaticy_Scale: Los colores inputeados tienen distintos canales.')

    def residual_function(p):
        I_c = I * p
        I_c_n = I_c / np.linalg.norm(I_c)
        M_n = M / np.linalg.norm(M)
        return np.sum((I_c_n - M_n) ** 2)

    scale = fmin(residual_function, np.ones(len(M)), disp=False, xtol=1e-12, ftol=1e-12)
    return scale
