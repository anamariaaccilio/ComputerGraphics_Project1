import numpy as np

def Reinhard_White_Point(L, delta=1e-6):
    LMin = np.min(L[L > 0])
    LMax = np.max(L[L > 0])

    log2Min = np.log2(LMin + delta)
    log2Max = np.log2(LMax + delta)

    wp = 1.5 * 2**(log2Max - log2Min - 5)
    
    return wp