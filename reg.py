import re

def polynomial(poly: str):
    # Extract variables from the polynomial
    variables = sorted(list(set(re.findall(r'[a-zA-Z]', poly))))

    # Replace '^' with '**' for Python's power operator
    poly = poly.replace('^', '**')
    
    # Create lambda function
    func = eval('lambda {}: {}'.format(','.join(variables), poly))
    
    return func, len(variables)


d = 2
x = [1,2]

polynomial = '2*x1^3 + 4*x1^2 -8*x1 + 3*x2^4 - 2*x2^2 - 5*x2 + x1*x2 + 8'

# separate all words except xi
tmp = ""
for i in polynomial:
    if i == "x":
        tmp += i
    else:
        tmp += f"{i} "
polynomial = tmp

# replace xi with x[i-1]
for i in range(d):
    polynomial = polynomial.replace(f'x{i+1}',f'x[{i}]')

# replace '^' with '**'
polynomial = polynomial.replace('^','**')

# defice function using eval
f = lambda x: eval(polynomial)

print(f([0,1]))
