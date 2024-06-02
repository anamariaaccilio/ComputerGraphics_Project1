import numpy as np
from Rgb_to_Srgb import Rgb_to_Srgb

def Remove_CRF(img, lin_type='gamma', lin_fun=2.2):
    lin_type = lin_type.lower()
    if lin_type == 'linear':
        return img  # Las imágenes son lineares por default.
    
    elif lin_type == 'gamma':
        return img ** lin_fun
    
    elif lin_type == 'srgb':
        return Rgb_to_Srgb(img, inverse=True)
    
    elif lin_type == 'lut':
        img_out = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_out[:, :, i] = np.take(lin_fun[:, i], img[:, :, i].astype(int))
        return img_out
    
    elif lin_type == 'poly':
        return np.polyval(lin_fun, img)
    
    else:
        raise ValueError(f"Tipo de linearización desconocida: {lin_type}")
