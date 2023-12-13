import numpy as np
import sympy
from scipy.optimize import line_search

class Optimizer:
    
    def __init__(self,equation):
        f, symbolic, x = self.polynomial(equation)
        self.f = f
        self.symbolic = symbolic
        self.x = x

    def inputModifier(self,fun):
        return lambda y : fun(*y)
    
    def polynomial(self,poly: str):

        varset = sorted(list(set(''.join(c for c in poly if c.isalpha()).lower())))
        vararr = [None for i in range(len(varset))]

        for i in range(len(varset)):
            vararr[i] = sympy.symbols(varset[i])
        
        symbolic = sympy.sympify(poly)
        func = sympy.lambdify(vararr, symbolic, "numpy")
        f = self.inputModifier(func)

        return f, symbolic, vararr

    def optimize(self,mode=0):

        x0 = np.random.rand(len(self.x),)
        grad = self.inputModifier(sympy.lambdify(self.x, sympy.Matrix([self.symbolic]).jacobian(self.x), "numpy"))
        d = lambda y : -1 * np.ravel(grad(y))
        
        if mode==1:
            hessianInverse = self.inputModifier(sympy.lambdify(self.x, sympy.hessian(self.symbolic, self.x)**-1, "numpy"))
            d = lambda y : -1 * np.ravel(hessianInverse(y) @ grad(y).T)
        
        e = 1e-6
        
        while True:
            t = line_search(self.f, grad, x0, d(x0))
            x1 = x0 + t[0] * d(x0)
            if np.abs(self.f(x1) - self.f(x0)) < e:
                break
            x0 = x1
        
        print(self.symbolic, '\nMinimum point: ', x1, '\nMinimal value: ', self.f(x1))

equation = '2*a**2 + 3*b**2 - 2*a*b + 2*a - 3*b'
optimizer = Optimizer(equation)
optimizer.optimize(0)