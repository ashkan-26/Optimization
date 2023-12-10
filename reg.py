import numpy as np
import sympy

def polynomial(poly: str):      # Input: polynomial function as a string

    # Finding all letters(variables), storing them in a set (for uniqueness), converting to a sorted array.
    varset = sorted(list(set(''.join(c for c in poly if c.isalpha()).lower())))

    # Creating an array to store each variable as a symbol.
    vararr = [None for i in range(len(varset))]
    for i in range(len(varset)):
        vararr[i] = sympy.symbols(varset[i])    # The variables will be addressed and initialized as the elements of this array.

    # Symbolic expression of polynomial.
    symbolic = sympy.sympify(poly)

    # Converting symbolic expression to a lambda function.
    f = sympy.lambdify(vararr, poly, "numpy")

    return f, symbolic, vararr
    # Output:
    # f = Polynomial function
    # symbolic = Symbolic notation of the polynomial. Will be used for computation of Gradient and Hessien matrices.
    # varrarr = List of polynomial variables.