import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

def Reinhard_Gaussian_Filter(img, s, alpha_i):
    
    alpha_s_sq = (alpha_i * s) ** 2

    r, c = img.shape[:2]
    x, y = np.meshgrid(np.arange(c) - c / 2, np.arange(r) - r / 2)
    kernel = np.exp(-(x**2 + y**2) / alpha_s_sq) / (np.pi * alpha_s_sq)
    kernel_f = fft2(kernel)

    imgOut = np.zeros_like(img)

    for i in range(img.shape[2]):
        imgOut[:, :, i] = fftshift(ifft2(fft2(img[:, :, i]) * kernel_f)).real

    return imgOut   # Botando la imagen tras el filtro
