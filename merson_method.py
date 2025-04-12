import numpy as np

def merson(y:np.ndarray,f,k:np.float64):
    tn = 0
    for i in range(len(y)-1):
        k1 = f(tn,y[i])
        k2 = f(tn+1/3*k,y[i]+1/3*k1)
        k3 =f(tn+1/3*k,y[i]+1/6*k1+1/6*k2)
        k4=f(tn+1/2*k,y[i]+1/8*k1+3/8*k3)
        k5=f(tn+k,y[i]+1/2*k1-3/2*k3+2*k4)

        y[i+1] = y[i] + k*(1/6*k1+2/3*k4+1/6*k5)
        tn += k
    
    return y