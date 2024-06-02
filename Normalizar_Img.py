import numpy as np

def Normalizar_Img(img, a, b):
    
    img_clamped = np.copy(img)
    img_clamped[img < a] = a
    img_clamped[img > b] = b
    return img_clamped