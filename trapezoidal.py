import jax.numpy as np
from jax import jacobian


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

def trapezoidal(F, y0: np.ndarray,pontos:int, t: np.ndarray, k: float, tol: float):
    num_equations = len(y0)
    y = np.empty((len(y0),pontos),dtype=float)
    y = y.at[:,0].set(y0[:])
    for i in range(pontos - 1):
        G = lambda x: np.array([(x[j] - y[j,i]) / k - (F[j](t[i+1], x[j]) + F[j](t[i], y[j,i])) / 2 for j in range(num_equations)]).T
        dG = jacobian(G)
        guest = y[:,i]
        while True:
            new_guest = guest - np.linalg.inv(dG(guest))@G(guest)
            if np.abs(new_guest[0] - guest[0]) <= tol:
                y = y.at[:,i+1].set(new_guest)
                break
            else:
                guest = new_guest
    return y