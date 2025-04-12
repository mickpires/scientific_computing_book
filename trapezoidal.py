import jax.numpy as np
from jax import grad


# trapezoidal usando ponto fixo

# def trapezoidal(f,y:np.ndarray,k:float,tol=float):
#     for i in range(len(y)-1):
#         guest = y[i]
#         while True:
#             new_guest = y[i] + k*(f(guest) + f(y[i]))/2
#             if np.abs(new_guest-guest) <= tol:
#                 y[i+1] = new_guest
#                 break
#             else:
#                 guest = new_guest
#     return y

def trapezoidal(F, y: np.ndarray, t: np.ndarray, k: float, tol: float):
    for i in range(len(y) - 1):
        G = lambda x: (x - y[i]) / k - (F(t[i+1], x) + F(t[i], y[i])) / 2
        dG = grad(G)
        guest = y[i]
        while True:
            new_guest = guest - G(guest) / dG(guest)
            if np.abs(new_guest - guest) <= tol:
                y[i+1] = new_guest
                break
            else:
                guest = new_guest
    return y