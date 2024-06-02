import numpy as np

from Lum import Lum
from Remove_Specials import Remove_Specials

def Change_Luminance(img, Lold, Lnew, bEpsilon=False):
    col = img.shape[2]
    col_new = 1 if len(Lnew.shape) == 2 else Lnew.shape[2]

    imgOut = np.zeros_like(img)

    if bEpsilon:
        Lold = Lold + 1e-6

    if col_new == 1:
        if col == col_new:
            imgOut = Lnew
        else:
            for i in range(col):
                imgOut[:, :, i] = (img[:, :, i] * Lnew) / Lold
    elif col_new == 3:
        if col == col_new:
            for i in range(col):
                imgOut[:, :, i] = (img[:, :, i] * Lnew[:, :, i]) / Lold
        else:
            Lnew = Lum(Lnew)
            for i in range(col):
                imgOut[:, :, i] = (img[:, :, i] * Lnew) / Lold
    else:
        Lnew = Lum(Lnew)
        for i in range(col):
            imgOut[:, :, i] = (img[:, :, i] * Lnew) / Lold

    imgOut = Remove_Specials(imgOut)

    return imgOut
