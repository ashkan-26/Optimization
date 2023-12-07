import re

# if __name__ == '__main__':

def polynomial(poly: str):
    # Extract variables from the polynomial
    variables = sorted(list(set(re.findall(r'[a-zA-Z]', poly))))

    # Replace '^' with '**' for Python's power operator
    poly = poly.replace('^', '**')
    
    # Create lambda function
    func = eval('lambda {}: {}'.format(','.join(variables), poly))
    
    return func, len(variables)