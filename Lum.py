import numpy as np

def Lum(img):
    if img.ndim == 3:
        col = img.shape[2]
    else:
        col = 1

    if col == 1:
        return img
    elif col == 3:
        return (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2])
    else:
        print("Se computó la media de los canales, pero el archivo no es ni RGB ni una imagen de luminancia.")
        return np.mean(img, axis=2)
