import numpy as np

from Weight_Function import Weight_Function
from LDR_Stack_Sub_Sampling import LDR_Stack_Sub_Sampling
from Gsolve import Gsolve
from Find_Chromaticy_Scale import Find_Chromaticy_Scale

def Debevec_CRF(stack, stack_exposure, n_samples=256, sampling_strategy='Grossberg', smoothing_term=128, b_normalize=True):
    if len(stack) == 0:
        raise ValueError('DebevecCRF: a stack cannot be empty!')

    if len(stack_exposure) == 0:
        raise ValueError('DebevecCRF: a stack_exposure cannot be empty!')

    if stack.shape[3] != len(stack_exposure):
        raise ValueError('stack and stack_exposure have different number of exposures')

    if stack.dtype == np.uint8:
        stack = stack.astype(np.float32) / 255.0

    if stack.dtype == np.uint16:
        stack = stack.astype(np.float32) / 65535.0

    col = stack.shape[2]

    W = Weight_Function(np.linspace(0, 1, 256), 'Deb97')

    stack_samples = LDR_Stack_Sub_Sampling(stack, stack_exposure, n_samples, sampling_strategy)

    # Recuperando el CRF
    lin_fun = np.zeros((256, col))
    log_stack_exposure = np.log(stack_exposure)

    max_lin_fun = np.zeros(col)

    for i in range(col):
        g = Gsolve(stack_samples[:, :, i], log_stack_exposure, smoothing_term, W)
        g = np.exp(g)

        lin_fun[:, i] = g

    # Corrección de colores.
    gray = np.array([lin_fun[128, i] for i in range(col)])

    scale = Find_Chromaticy_Scale([0.5, 0.5, 0.5], gray)

    for i in range(col):
        lin_fun[:, i] = scale[i] * lin_fun[:, i]
        max_lin_fun[i] = np.max(g)

    if b_normalize:
        max_val = np.max(max_lin_fun)

        for i in range(col):
            lin_fun[:, i] /= max_val

    return lin_fun, max_lin_fun
