import numpy as np
import matplotlib.pyplot as plt

from Normalizar_Img import Normalizar_Img

def Gamma_TMO(img, TMO_gamma=2.2, TMO_fstop=0.0, TMO_view=False):

    if np.any(img < 0):
        raise ValueError('La imagen contiene valores negativos.')

    if TMO_gamma <= 0.0:
        raise ValueError('TMO_gamma debe ser un valor positivo escalar.')

    invGamma = 1.0 / TMO_gamma
    exposure = 2.0 ** TMO_fstop

    # Normalizando los valores entre [0.0, 1.0]
    imgOut = Normalizar_Img((exposure * img) ** invGamma, 0, 1)

    if TMO_view:
        plt.imshow(imgOut)
        plt.axis('off')
        plt.show()

    return imgOut # Gamma corrected exposure
