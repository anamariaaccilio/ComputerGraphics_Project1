import numpy as np

def Log_Mean(img, delta=1e-6):

    img_delta = np.log(img + delta)

    Lav = np.exp(np.mean(img_delta))

    return Lav
