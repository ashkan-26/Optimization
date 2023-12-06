import numpy as np
import sympy as sym
from numdifftools import Gradient
from scipy.optimize import minimize_scalar

# Showing functions and computing its gradiant symbolically
x, y = sym.symbols('x, y')
f = 8 * x**2 + 4 * y ** 2 - 9
Df = sym.Matrix([f]).jacobian(sym.Matrix(list(f.free_symbols)))

# Basic Algorithm
f = lambda x : x[0]**2 + x[1]**2 - 1
x0 = np.random.rand(2,) * 10
d = lambda x : Gradient(f)(x)
e = 1e-5
# t = minimize_scalar(lambda alpha: f(x0 - alpha * d(x0))).x
t = 0.5
while True:
    x1 = x0 - t*d(x0)
    if np.abs(f(x1) - f(x0)) < e:
        break
    x0 = x1

print(x0)
print(t)
print(f(x0))