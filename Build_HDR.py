import numpy as np
from scipy.ndimage import gaussian_filter
from math import log, exp

from Debevec_CRF import Debevec_CRF
from Normalizar_Img import Normalizar_Img
from Weight_Function import Weight_Function
from Remove_CRF import Remove_CRF


def Build_HDR(stack, stack_exposure, lin_type='gamma', lin_fun=None, weight_type='all', merge_type='log', b_mean_weight=False):
    # Check input arguments
    if stack is None:
        raise ValueError('El stack está vacío!')

    if stack_exposure is None:
        raise ValueError('El stack_exposure está vacío!')

    stack_exposure_check = np.unique(stack_exposure)
    if len(stack_exposure) != len(stack_exposure_check):
        raise ValueError('El stack contiene imágenes con el mismo valor de exposición. Borra los duplicados.!')
    if np.min(stack_exposure) <= 0.0:
        raise ValueError('El stack contiene imágenes con un valor de exposición negativo. Remueve estas imágenes.')

    # Initialize output images
    r, c, col, n = stack.shape
    img_out = np.zeros((r, c, col), dtype=np.float32)
    tot_weight = np.zeros((r, c, col), dtype=np.float32)

    scale = 1.0
    if stack.dtype == np.uint8:
        scale = 255.0
    elif stack.dtype == np.uint16:
        scale = 65535.0

    if lin_type == 'LUT' and lin_fun is None:
        lin_fun, _ = Debevec_CRF(stack.astype(np.float32) / scale, stack_exposure)

    if lin_type == 'gamma':
        if lin_fun is None or lin_fun <= 0.0:
            lin_fun = 2.2

    delta_value = 1.0 / 65535.0

    i_sat = np.argmin(stack_exposure)
    i_noisy = np.argmax(stack_exposure)
    threshold = 0.9

    for i in range(n):
        tmp_stack = Normalizar_Img(stack[:, :, :, i].astype(np.float32) / scale, 0.0, 1.0)

        weight_type_i = weight_type
        if i == i_sat and np.any(tmp_stack > threshold):
            weight_type_i = 'identity'
        if i == i_noisy and np.any(tmp_stack < 1.0 - threshold):
            weight_type_i = 'reverse'

        weight = Weight_Function(tmp_stack, weight_type_i, b_mean_weight)
        weight[tmp_stack <= delta_value] = 0.0

        tmp_stack = Remove_CRF(tmp_stack, lin_type, lin_fun)
        dt_i = stack_exposure[i]

        if merge_type == 'linear':
            img_out += (weight * tmp_stack) / dt_i
            tot_weight += weight
        elif merge_type == 'log':
            img_out += weight * (np.log(tmp_stack + delta_value) - np.log(dt_i))
            tot_weight += weight
        elif merge_type == 'w_time_sq':
            img_out += (weight * tmp_stack) * dt_i
            tot_weight += weight * dt_i * dt_i

    img_out = img_out / tot_weight
    if merge_type == 'log':
        img_out = np.exp(img_out)

    saturation = 1e-4
    if np.any(tot_weight <= saturation):
        i_med = len(stack_exposure) // 2
        med = Normalizar_Img(stack[:, :, :, i_med].astype(np.float32) / scale, 0.0, 1.0)
        tmp_stack = Normalizar_Img(stack[:, :, :, i_sat].astype(np.float32) / scale, 0.0, 1.0)
        img_sat = Remove_CRF(tmp_stack, lin_type, lin_fun) / stack_exposure[i_sat]

        mask = np.zeros_like(tot_weight)
        mask[tot_weight <= saturation & (med > 0.5)] = 1
        mask = np.max(mask, axis=2)
        if np.max(mask) > 0.5:
            for i in range(col):
                io_i = img_out[:, :, i]
                is_i = img_sat[:, :, i]
                io_i[mask == 1] = is_i[mask == 1]
                img_out[:, :, i] = io_i

        tmp_stack = Normalizar_Img(stack[:, :, :, i_noisy].astype(np.float32) / scale, 0.0, 1.0)
        img_noisy = Remove_CRF(tmp_stack, lin_type, lin_fun) / stack_exposure[i_noisy]

        mask = np.zeros_like(tot_weight)
        mask[tot_weight <= saturation & (med < 0.5)] = 1
        mask = np.max(mask, axis=2)
        if np.max(mask) > 0.5:
            for i in range(col):
                io_i = img_out[:, :, i]
                in_i = img_noisy[:, :, i]
                io_i[mask == 1] = in_i[mask == 1]
                img_out[:, :, i] = io_i

    return img_out, lin_fun
