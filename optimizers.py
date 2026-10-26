import numpy as np

class AdamOptimizer:
    def __init__(self, beta1=0.99, beta2=0.999):
  
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None  
        self.v = None  
        self.t = 0    

    def update(self, learning_rate, params, grads):
       
        if self.m is None:
            self.m = np.zeros_like(params)
        if self.v is None:
            self.v = np.zeros_like(params)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        params -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        return params
    

class RMSpropOptimizer:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.grad_squared = None  

    def update(self, learning_rate, params, grads):
        if self.grad_squared is None:
            self.grad_squared = np.zeros_like(grads)

        self.grad_squared = self.beta * self.grad_squared + (1 - self.beta) * (grads ** 2)
        params -= learning_rate * grads / (np.sqrt(self.grad_squared) + 1e-8)

        return params