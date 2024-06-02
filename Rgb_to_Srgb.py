import numpy as np

def Rgb_to_Srgb(img, inverse=False):
    if inverse:
        return np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1 / 2.4)) - 0.055)
    else:
        return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)