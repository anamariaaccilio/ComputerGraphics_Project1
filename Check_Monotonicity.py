import numpy as np

def Check_Monotonicity(sort_index, values):
    check = 0
    n = len(sort_index) - 1
    col = values.shape[1]

    # Reordenando los valores por su index
    values = values[sort_index, :]

    for j in range(col):
        val = 1
        for i in range(n):
            if values[i, j] <= values[i + 1, j]:
                val = 0
                break
        check += val

    return check
