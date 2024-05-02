import numpy as np

class logdouble:

    log_val = -1
    original_val = -1

    def __init__(self, x, log=False):
        if log == True:
            self.log_val = x
            self.original_val = np.exp(self.log_val)
        else:
            self.original_val = x
            self.log_val = np.log(x)
        

    def __mul__(self, other):
        prod = logdouble(self.log_val + other.log_val, 1)
        return prod
    
    def __truediv__(self, other):
        div = logdouble(self.log_val - other.log_val, 1)
        return div
    
    def __add__(self, other):
        sum = 0.0
        if other.log_val <= self.log_val:
            sum = self.log_val + np.log(1 + np.exp(other.log_val - self.log_val))
        else:
            sum = other.log_val + np.log(1 + np.exp(self.log_val - other.log_val))
        
        sum = logdouble(sum, True)
        return sum
    
    def __str__(self):
        return str(self.log_val)
    
    def o(self):
        return self.original_val
    def e(self):
        return self.log_val
    