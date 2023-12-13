import numpy as np
import sympy
from scipy.optimize import line_search
from optim import *

polynomial = 'a**4 + 2*a**2*b + b**2 - 4*a**2 - 8*a - 8*b'
polynomial = Polynomial(polynomial)
f = polynomial.f

# Algorithm

x0 = np.random.rand(len(polynomial.x),) * 10

gradient_symbolic = sympy.Matrix([polynomial.symbolic]).jacobian(polynomial.x)
gradient = inputModifier(sympy.lambdify(polynomial.x, gradient_symbolic, "numpy"))
hessianInverse = inputModifier(sympy.lambdify(polynomial.x, sympy.hessian(polynomial.symbolic, polynomial.x)**-1, "numpy"))

d = lambda y : -1 * np.ravel(hessianInverse(y) @ gradient(y).T)

e = 1e-6

while True:
    t = line_search(f, gradient, x0, d(x0))
    x1 = x0 + t[0] * d(x0)
    if np.abs(f(x1) - f(x0)) < e:
        break
    x0 = x1

print(polynomial.symbolic, '\nMinimum point: ', x1, '\nMinimal value: ', f(x1))