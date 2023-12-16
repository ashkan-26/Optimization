import numpy as np
import sympy
from scipy.optimize import line_search
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()     # Set Seaborn as main theme for plots


# Define a class for optimizing equations
class Optimizer:
    
    def __init__(self, equation):
        f, symbolic, x = self.polynomial(equation)
        self.e = 1e-5
        self.f = f
        self.symbolic = symbolic
        self.x = x

    def inputModifier(self, fun):
        return lambda y: fun(*y) 

    def polynomial(self, poly: str):
        varset = sorted(list(set(''.join(c for c in poly if c.isalpha()).lower())))
        vararr = [None for i in range(len(varset))]

        for i in range(len(varset)):
            vararr[i] = sympy.symbols(varset[i])
        
        symbolic = sympy.sympify(poly)
        func = sympy.lambdify(vararr, symbolic, "numpy")
        f = self.inputModifier(func)
        
        return f, symbolic, vararr

    def plot_equation(self):
        a_vals = np.linspace(-10, 10, 100)

        if len(self.x) == 1: 
            y_values = [self.f([float(x)]) for x in a_vals]
            self.fig , self.ax = plt.subplots()
            self.ax.plot(a_vals,y_values)

        elif len(self.x) == 2:
            b_vals = np.linspace(-10, 10, 100)
            A, B = np.meshgrid(a_vals, b_vals)
            Z = self.f([A, B])
            
            self.fig = plt.figure("Optimization")
            self.ax = plt.axes(projection='3d',xlabel='a')
            self.ax.plot_surface(A, B, Z, rstride=1, cstride=1,cmap='Blues', edgecolor = 'none')
        self.ax.set_title(self.symbolic)

    def scatter(self,x0):
        if len(self.x) == 1:
            self.ax.scatter(x0[0],self.f(x0),c='red',s=50,marker='*')
        elif len(self.x) == 2:
            self.ax.scatter(x0[0],x0[1],self.f([x0[0],x0[1]]),c='red',s=50,marker='*')

    def optimize(self, mode=0):

        x0 = np.random.rand(len(self.x),)
        grad = self.inputModifier(sympy.lambdify(self.x, sympy.Matrix([self.symbolic]).jacobian(self.x), "numpy"))
        
        self.scatter(x0)

        if mode==0:
            d = lambda y: np.ravel(grad(y))
            while True:
                
                plt.pause(1) 
                t = line_search(self.f, grad, x0, -1 * d(x0))

                try:
                    x1 = x0 - t[0] * d(x0)
                except:
                    print("not convergant")
                    x0 *= (-1)
                    continue

                if np.abs(self.f(x1) - self.f(x0)) < self.e:
                    break
                
                x0 = x1
                
                self.scatter(x1)
                

        else:
            gradient_symbolic = sympy.Matrix([self.symbolic]).jacobian(self.x)
            hessianInverse = self.inputModifier(sympy.lambdify(self.x, sympy.hessian(self.symbolic, self.x)**-1, "numpy"))
            d = lambda y : -1 * np.ravel(hessianInverse(y) @ grad(y).T)

            while True:
                
                plt.pause(1) 
                t = line_search(self.f, grad, x0, d(x0))

                try:
                    x1 = x0 + t[0] * d(x0)
                except:
                    print("not convergant")
                    x0 *= (-1)
                    continue

                if np.abs(self.f(x1) - self.f(x0)) < self.e:
                    break

                x0 = x1

                self.scatter(x1)

        print(self.symbolic, '\nMinimum point: ', x1, '\nMinimal value: ', self.f(x1))
        plt.show()


    
        

# equation = 'a**4 + 2*a**2*b + b**2 -4*a**2 -8*a -8*b'
# equation = 'x**2 + y**2 - 10'
# equation= '100*x**4 - 0.01*y**4'
# equation = '(x**2 + 1)**0.5 + (y**2 + 1)**0.5'
# equation = '(x**2 + 1)**0.5'
equation = '(x**2 + 1)**0.5 + (y**2 + 1)**0.5'
        
optimizer = Optimizer(equation)
optimizer.plot_equation()
optimizer.optimize(1)