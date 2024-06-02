import numpy as np
from scipy.stats import norm

from LDR_Stack_Histogram import LDR_Stack_Histogram
from Grossberg_Sampling import Grossberg_Sampling
from Spatial_Sampling import Spatial_Sampling

def LDR_Stack_Sub_Sampling(stack, stack_exposure, n_samples=256, sampling_strategy='Grossberg', outliers_percentage=0):
    if stack.size == 0:
        raise ValueError('El stack está vacío.')

    sort_index = np.argsort(stack_exposure)[::-1]

    if sampling_strategy == 'Grossberg':
        stack_hist = LDR_Stack_Histogram(stack)
        stack_samples = Grossberg_Sampling(stack_hist, n_samples)
    elif sampling_strategy in ['RandomSpatial', 'RegularSpatial']:
        stack_samples = Spatial_Sampling(stack, sort_index, n_samples, sampling_strategy)
        stack_samples = np.round(stack_samples * 255)
    else:
        raise ValueError('LDRStackSubSampling: No se especificó la estrategia de sampleo.')

    if outliers_percentage > 0.0:
        t_min = outliers_percentage
        t_max = 1.0 - t_min
        stack_samples[(stack_samples < (t_min * 255.0)) | (stack_samples > (t_max * 255))] = -1.0
    
    return stack_samples
