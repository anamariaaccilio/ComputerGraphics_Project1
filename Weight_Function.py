import numpy as np

def Weight_Function(img, weight_type, b_mean_weight=False, bounds=[0, 1], pp=None):
    if len(img.shape) == 3 and img.shape[2] > 1 and b_mean_weight:
        img = np.mean(img, axis=2, keepdims=True)
    
    if weight_type == 'Deb97_p05':
        bounds = [0.05, 0.95]
        weight_type = 'Deb97'
    
    if weight_type == 'all': weight = np.ones_like(img)
    elif weight_type == 'identity': weight = img
    elif weight_type == 'reverse': weight = 1.0 - img
    elif weight_type == 'box':
        weight = np.ones_like(img)
        weight[(img < bounds[0]) | (img > bounds[1])] = 0.0
    elif weight_type == 'Robertson':
        shift = np.exp(-4)
        scale_div = 1.0 - shift
        t = img - 0.5
        weight = (np.exp(-16.0 * (t ** 2)) - shift) / scale_div
    elif weight_type == 'hat': weight = 1 - (2 * img - 1) ** 12
    elif weight_type == 'poly':
        weight = np.zeros_like(img)
        for i in range(img.shape[2]):
            d_pp = np.polyder(pp[:, i])
            weight[:, :, i] = np.polyval(pp[:, i], img) / np.polyval(d_pp, img)
    elif weight_type == 'Deb97':
        Zmin, Zmax = bounds
        tr = (Zmin + Zmax) / 2
        delta = Zmax - Zmin
        
        weight = np.zeros_like(img)
        indx1 = img <= tr
        indx2 = img > tr
        weight[indx1] = img[indx1] - Zmin
        weight[indx2] = Zmax - img[indx2]
        
        if delta > 0.0: weight /= tr
    else:
        weight = -1
    
    weight = np.clip(weight, 0.0, 1.0)
    return weight