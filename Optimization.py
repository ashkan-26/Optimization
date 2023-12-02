import numpy as np
import sympy as sym
from scipy.optimize import line_search

# Showing functions and computing its gradiant symbolically
x, y = sym.symbols('x, y')
f = 8 * x**2 + 4 * y ** 2 - 9
Df = np.array(sym.Matrix([f]).jacobian(sym.Matrix(list(f.free_symbols))))

# Finding the best alpha value using scipy.optimize.line_search: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
f = lambda x : 8*(x[0]**2) + 4*(x[1]**2) - 9    # Function
Df = lambda x : np.array([16*x[0], 8*x[1]])     # Gradient of the function
x0 = np.array([2, 4])                           # Initial value
d = Df(x0) * -1                                 # Descent direction(In Gradient method, it is set to -∇f(x) )
search = line_search(f, Df, x0, d)
alpha = search[0]
print(alpha)