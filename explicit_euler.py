import numpy as np

def explicit_euler(f,y:np.ndarray,k:float):
    for i in range(len(y)-1):
        y[i+1] = y[i] +k*f(y[i])
    return y