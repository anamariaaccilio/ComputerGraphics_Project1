import numpy as np

def Remove_Specials(img, clamping_value=0):
    img[np.isnan(img) | np.isinf(img)] = clamping_value
    return img