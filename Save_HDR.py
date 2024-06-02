import cv2
import numpy as np

def Save_HDR(filename, imgHDR):     # Guarda una imagen HDR en un archivo con formato .hdr.
    if imgHDR.dtype != np.float32:
        imgHDR = imgHDR.astype(np.float32)
    cv2.imwrite(filename, imgHDR)
