
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn




"""
Values for the hyperparameters
"""

a = 10
s = 0.1
l = 1
dt = 0.1
max_it = 200

N = 10000
n = 1000
d = 100




"""
Let's define our first function f and its gradients
"""
def f(X):
    alpha = 10**(-1)*torch.ones_like(X)
    x = torch.matmul(X.t(), alpha)
    
    return ((x - 5)**2 * (2 + torch.cos(x * 2 * torch.tensor(np.pi))))


def grad_f(X):
    X.requires_grad_(True) 
    Y = f(X)  
    gradients = torch.autograd.grad(Y, X)[0] 
    return gradients


X = torch.linspace(0, 100, 100).unsqueeze(1)
Y = torch.stack([f(x) for x in X])

# Plot the figure
plt.figure()
plt.plot(X.numpy(), Y.numpy())  
plt.show()


# 1. Using CBO by hand !



"""
First we develop our own method to generate the dynamic of the CBO method
"""
def euler_maruyama_step(X0, dt, a, l, s,f):
    X = torch.zeros_like(X0)
    n = X.shape[0]
    w=torch.stack([torch.exp(-a*f(x)) for x in X0]).unsqueeze(1)
    mt = torch.sum(X0*w,dim=0) / torch.sum(w)
    dW = torch.randn(n).unsqueeze(1)
    X = X0 - l * (X0 - mt.expand(N, -1)) * dt + s * torch.abs(X0 - mt.expand(N, -1)) * dW
    return X

     

# Initialization
X0 = torch.randn(N, d)

# Euler Maruyama method
for i in range(max_it+1):
    X = euler_maruyama_step(X0, dt, a, l, s,f)
    X0 = X.clone()

    if i % 50 == 0:
        w=torch.stack([torch.exp(-a*f(x)) for x in X0]).unsqueeze(1)
        mt = torch.sum(X0*w,dim=0) / torch.sum(w)
        alpha = torch.ones_like(mt)
        x = torch.matmul(mt.T, alpha)
        print(30*'-')
        print('Step ',i)
        print('value of the function : ',f(mt).item())
        print('value of the projection : ', x.item())
        print(30*'-')


# 2. Using CBO with the python library CBX



import numpy as np
import cbx
import matplotlib.pyplot as plt

"""
define the objective function and solve
"""

def f(x):
    return np.abs(np.sin(x)) + np.abs(x)**(3/4) * (np.sin(x)+1) - 0.5*(np.abs(x)>1)

dyn = cbx.dynamics.CBO(f, N=50,d=1, verbosity=0)
for i in range(1) :
    dyn.step()
x=dyn.x

plt.close('all')
s = np.linspace(-4,4,1000)
plt.plot(s, f(s), linewidth=3, color='xkcd:sky', label='Objective', zorder=-1)
plt.scatter(x, f(x), label='Solution', c='green', s=50, marker='x')
plt.xlim([-4,4])
plt.legend()

