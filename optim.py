# import numpy as np
# import sympy

# def inputModifier(fun):     # Input: function with multimple output
    
#     return lambda y : fun(*y)

#     '''Output: function with one input'''

# def polynomial(poly: str):      # Input: polynomial function as a string

#     # Finding all letters(variables), storing them in a set (for uniqueness), converting to a sorted array.
#     varset = sorted(list(set(''.join(c for c in poly if c.isalpha()).lower())))

#     # Creating an array to store each variable as a symbol.
#     vararr = [None for i in range(len(varset))]
#     for i in range(len(varset)):
#         vararr[i] = sympy.symbols(varset[i])    # The variables will be addressed and initialized as the elements of this array.

#     # Symbolic expression of polynomial.
#     symbolic = sympy.sympify(poly)

#     # Converting symbolic expression to a lambda function.
#     func = sympy.lambdify(vararr, poly, "numpy")
#     f = inputModifier(func)

#     return f, symbolic, vararr

#     ''' 
#     Output:
#     f = Polynomial function
#     symbolic = Symbolic notation of the polynomial. Will be used for computation of Gradient and Hessien matrices.
#     varrarr = List of polynomial variables.
#     '''

# # def Gradient(poly): 

# #     _, symbolic, x = polynomial(poly)

# #     return inputModifier(x, sympy.Matrix([symbolic]).jacobian(x), "numpy")



import numpy as np
import sympy
from scipy.optimize import line_search

class Optimizer:
    
    def __init__(self,equation):
        f, symbolic, x = self.polynomial(equation)
        self.e = 1e-5
        self.f = f
        self.symbolic = symbolic
        self.x = x

    def inputModifier(self,fun):
        return lambda y : fun(*y) 
    
    def polynomial(self,poly: str):      # Input: polynomial function as a string

        # Finding all letters(variables), storing them in a set (for uniqueness), converting to a sorted array.
        varset = sorted(list(set(''.join(c for c in poly if c.isalpha()).lower())))

        # Creating an array to store each variable as a symbol.
        vararr = [None for i in range(len(varset))]
        for i in range(len(varset)):
            vararr[i] = sympy.symbols(varset[i])    # The variables will be addressed and initialized as the elements of this array.

        # Symbolic expression of polynomial.
        symbolic = sympy.sympify(poly)

        # Converting symbolic expression to a lambda function.
        func = sympy.lambdify(vararr, symbolic, "numpy")
        f = self.inputModifier(func)

        return f, symbolic, vararr

    def optimize(self,mode=0):
        x0 = np.random.rand(len(self.x),)
        grad = self.inputModifier(sympy.lambdify(self.x, sympy.Matrix([self.symbolic]).jacobian(self.x), "numpy"))
        if mode==1:
            grad = sympy.Matrix([sympy.Matrix([self.symbolic]).jacobian(self.x)]).jacobian(self.x)
        
        d = lambda y : np.ravel(grad(y))
        
        while True:
            t = line_search(self.f, grad, x0, -1*d(x0))
            x1 = x0 - t[0] * d(x0)
            if np.abs(self.f(x1) - self.f(x0)) < self.e:
                break
            x0 = x1
        
        print(self.symbolic, '\nMinimum point: ', x1, '\nMinimal value: ', self.f(x1))

equation = '2*b**3 - 6*b**2 + 3*a**2*b'
optimizer = Optimizer(equation)
optimizer.optimize()