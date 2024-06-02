import numpy as np
from Lum import Lum
from Reinhard_Alpha import Reinhard_Alpha
from Reinhard_White_Point import Reinhard_White_Point
from Reinhard_Filtering import Reinhard_Filtering
# import Reinhard_Bilateral_Filtering
# import Filter_Gaussian
from Change_Luminance import Change_Luminance

def Reinhard_TMO(img, pAlpha=None, pWhite=None, pLocal='global', pPhi=8, Lwa=None):
    # Check if the image is grayscale or RGB
    if img.ndim == 3:
        col = img.shape[2]
    else:
        col = 1

    if col not in [1, 3]:
        raise ValueError('The image has to be an RGB or luminance image.')

    L = Lum(img)

    # Setting default values.
    if pAlpha is None or pAlpha <= 0:
        pAlpha = Reinhard_Alpha(L)
    
    if pWhite is None or pWhite <= 0:
        pWhite = Reinhard_White_Point(L)
    
    if pPhi < 0:
        pPhi = 8

    if Lwa is None or Lwa < 0.0:
        img_delta = np.log(img + 1e-6)  # Usando un delta default.
        Lwa = np.exp(np.mean(img_delta))

    # Scale luminance using alpha and logarithmic mean
    Lscaled = (pAlpha * L) / Lwa

    # compute adaptation
    if pLocal == 'local':
        L_adapt = Reinhard_Filtering(Lscaled, pAlpha, pPhi)
    elif pLocal == 'bilateral': pass
        # L_adapt = Reinhard_Bilateral_Filtering(Lscaled, pAlpha, pPhi)
    elif pLocal == 'mean': pass
        # L_adapt = Filter_Gaussian(Lscaled, pPhi)
    else:   # Incluye pLocal = 'global'
        L_adapt = Lscaled

    # Range compression
    Ld = (Lscaled * (1 + Lscaled / pWhite**2)) / (1 + L_adapt)

    # change luminance
    imgOut = Change_Luminance(img, L, Ld)
    
    return imgOut, pAlpha, pWhite
