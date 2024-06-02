import numpy as np
from Normalizar_Img import Normalizar_Img

def Avg_Luminance(exposure_time, aperture_value=1.0, iso_value=1.0, K_value=12.5):
    K_value = Normalizar_Img(K_value, 10.6, 13.4)

    value = (K_value * aperture_value ** 2) / (iso_value * exposure_time)
    value_inv = (iso_value * exposure_time) / (K_value * aperture_value ** 2)

    return value, value_inv
