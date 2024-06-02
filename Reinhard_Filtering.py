import numpy as np
from Reinhard_Alpha import Reinhard_Alpha
from Reinhard_Gaussian_Filter import Reinhard_Gaussian_Filter

def Reinhard_Filtering(L, pAlpha=None, pPhi=8, pEpsilon=0.05):    
    if pAlpha is None:
        pAlpha = Reinhard_Alpha(L)

    sMax = 8
    r, c = L.shape
    V_vec = np.zeros((r, c, sMax))

    alpha1 = 1 / (2 * np.sqrt(2))
    constant = (2 ** pPhi) * pAlpha

    s = 1
    for i in range(sMax):
        V_vec[:, :, i] = Reinhard_Gaussian_Filter(L, s, alpha1)
        s *= 1.6

    L_adapt = V_vec[:, :, sMax - 1]
    mask = np.zeros((r, c))
    for i in range(sMax - 1):
        V1 = V_vec[:, :, i]
        V2 = V_vec[:, :, i + 1]

        V = np.abs((V1 - V2) / ((constant / (s ** 2)) + V1))

        indx = np.where((V > pEpsilon) & (mask < 0.5))
        if indx[0].size > 0:
            mask[indx] = i + 1
            L_adapt[mask == (i + 1)] = V1[mask == (i + 1)]

    return L_adapt