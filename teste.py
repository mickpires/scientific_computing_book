from jax import jacobian
import jax.numpy as np

def f(x):
    return x**2

x = np.array([[2.]])
df = jacobian(f)
x_flat = x.flatten()

print(df(x_flat))