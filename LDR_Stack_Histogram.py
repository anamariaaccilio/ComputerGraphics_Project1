import numpy as np
from skimage.util import img_as_ubyte

def LDR_Stack_Histogram(stack):
    _, _, col, n = stack.shape
    stack_out = np.zeros((256, col, n), dtype=int)

    for i in range(n):
        for j in range(col):
            tmp = stack[:, :, j, i]

            if tmp.dtype in [np.float32, np.float64]:
                tmp = img_as_ubyte(tmp)
            
            if tmp.dtype == np.uint16:
                tmp = np.clip(np.round(tmp / 255.0), 0, 255).astype(np.uint8)
                print('Parece ser que la imagen no es de 16-bits. El máximo ha sido seteado a 65535.')
            
            # Usar np.histogram con 256 bins
            hist, _ = np.histogram(tmp, bins=256, range=(0, 255))
            stack_out[:, j, i] = hist
    
    return stack_out
