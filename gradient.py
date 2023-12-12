import numpy as np
import sympy
from scipy.optimize import line_search
from optim import polynomial, inputModifier

f = '2*b**3 - 6*b**2 + 3*a**2*b'
f, symbolic, x = polynomial(f)

# Algorithm

x0 = np.random.rand(len(x),)    # Change the range of x0 by multiplying with different numbers(e.g. *10 or *-1) in the case of non-convergency.
grad_lambdify = sympy.lambdify(x, sympy.Matrix([symbolic]).jacobian(x), "numpy")
grad = inputModifier(grad_lambdify)
d = lambda y : np.ravel(grad(y))
e = 1e-5    # Will be initialized by user.
while True:
    t = line_search(f, grad, x0, -1*d(x0))
    x1 = x0 - t[0] * d(x0)
    if np.abs(f(x1) - f(x0)) < e:
        break
    x0 = x1

print(symbolic, '\nMinimum point: ', x1, '\nMinimal value: ', f(x1))