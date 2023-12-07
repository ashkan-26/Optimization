from reg import polynomial
import numpy as np
from numdifftools import Gradient
from scipy.optimize import minimize_scalar

f = 'a^4 - 2*a^2 + b^2 + 2*b*c + 2*c^2'

f, v = polynomial(f)

# Basic Algorithm
x0 = np.random.rand(v,) * 10
d = lambda x : Gradient(f)(*x)
e = 1e-5
t = 0.00005
while True:
    x1 = x0 - t*d(x0)
    if np.abs(f(*x1) - f(*x0)) < e:
        break
    x0 = x1

print('Minimum point: ', x1)
print('Minimal value', f(*x1))