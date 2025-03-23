import numpy as np

def leapfrog(f,y:np.ndarray,k:float):
    y[1] = y[0] + k*f(y[0])
    for i in range(1,len(y)-1):
        y[i+1] = y[i-1] +2*k*f(y[i])
    return y