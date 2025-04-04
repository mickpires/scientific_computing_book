import numpy as np

def implicit_euler(f,y:np.ndarray,k:float,tol=float):
    for i in range(len(y)-1):
        guest = y[i]
        while True:
            new_guest = y[i] + k*f(guest)
            if np.abs(new_guest-guest) <= tol:
                y[i+1] = new_guest
                break
            else:
                guest = new_guest
    return y