import numpy as np
from Max_Quart import Max_Quart
from Log_Mean import Log_Mean

def Reinhard_Alpha(L, delta=1e-6):
    LMin = Max_Quart(L, 0.01)
    LMax = Max_Quart(L, 0.99)

    log2Min = np.log2(LMin + delta)
    log2Max = np.log2(LMax + delta)
    logAverage = Log_Mean(L)
    log2Average = np.log2(logAverage + delta)

    alpha = 0.18 * 4**((2.0 * log2Average - log2Min - log2Max) / (log2Max - log2Min))
    
    return alpha