import numpy as np
from numdifftools import Gradient

equation = 'x1^2 +x2^2 -1'

indice = []     # list of all index values to find number of variables
constant = 0    # constant value of equation

for f in equation.split(" "):
    # check if f is constant value
    try:        
        var = f.split('x')
        index = int(var[1].split("^")[0])
        indice.append(index)
    except:
        constant = float(f)


size = max(indice)      # find the number of variables

eq = np.zeros((2,size),float)       # first row keeps coefficients, second row keeps powers

for f in equation.split(" "):
    # break if f is constant value
    if 'x' not in f:
        break
    var = f.split('x')
    index = int(var[1].split("^")[0])   # save index of selected part
    power = var[1].split("^")[1]        # save power of selected part
    # if coefficient is '' or '+', convert it to 1
    if var[0] in ['','+']:
        eq[0,index-1] = 1
    # if coefficient is '-', convert it to -1 
    elif var[0] == '-':
        eq[0,index-1] = -1
    else:
        eq[0,index-1] = float(var[0])
    eq[1,index-1] = power

def f(x,eq):
    ans = 0
    if len(x) != eq.shape[1]:
        return False
    for i in range(len(x)):
        ans += eq[0,i]*(x[i]**eq[1,i])
    ans += constant
    return ans

print(f([0,0],eq))
