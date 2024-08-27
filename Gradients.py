
import cbx as cbx
from cbx.dynamics.cbo import CBO

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from functorch import jacrev


from cbx.noise import anisotropic_noise
import cbx.utils.resampling as rsmp

import time




device = 'cuda' if torch.cuda.is_available() else 'cpu'



"""
Objective function
"""

def gtilde(x) :
    return( (x - 5)**2 * (2 + np.cos(x * 2 *(np.pi))))

def g(X):
    alpha = 10**(-1)*torch.ones_like(X)
    x = torch.matmul(X.t(), alpha)
    
    return (x - 5)**2 * (2 + torch.cos(x * 2 * torch.tensor(np.pi)))




def grad(x,f):
    x = x.clone().detach().requires_grad_(True) 
    y = f(x)
    y.backward(create_graph=True)
    gradient = x.grad
    return(gradient)


def gradf(x,f):
    jacobian_fn = jacrev(f)
    grads = jacobian_fn(x)
    return(grads)




x=np.linspace(-20,20,100)
y=gtilde(x)
plt.figure()
plt.plot(x,y,c='r')
plt.title('Objective function in 1D')
plt.show()


n = 10000
d = 500 # Dimensionality of R^d

# Generate random data
X = torch.rand(n, d)
Y = torch.tensor([g(x) for x in X])
grads = torch.stack([gradf(x,g) for x in X])

# Optionally, split into training and validation sets
split_ratio = 0.8
split_index = int(n * split_ratio)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = Y[:split_index], Y[split_index:]
grads_train, grads_val = grads[:split_index], grads[split_index:]

class CustomDatasetGrads(Dataset):
    def __init__(self, x, y, grad_f) :
        self.x = x
        self.y = y
        self.grad_f = grad_f

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        grads = torch.tensor(self.grad_f[idx], dtype=torch.float32)
        return X , y , grads

train_dataset = CustomDatasetGrads(X_train, y_train, grads_train)
val_dataset = CustomDatasetGrads(X_val, y_val, grads_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



"""
Our original neural network
"""

class Perceptron(nn.Module):
    def __init__(self, mean = 0.0, std = 1.0, 
                 act_fun=nn.ReLU,
                 sizes = None):
        super(Perceptron, self).__init__()
        #
        self.mean = mean
        self.std = std
        self.act_fun = act_fun()
        self.sizes = sizes if sizes else [100, 1]
        self.linears = nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(len(self.sizes)-1)])

    def __call__(self, x):
        x = x.view([x.shape[0], -1])
        x = (x - self.mean)/self.std
        
        for linear in self.linears:
            x = linear(x)
            x = self.act_fun(x)
        return x


"""
Our new neural network will have two inputs : the predicted value and the predicted gradients
"""
class Perceptron_Grads(nn.Module):
    def __init__(self, mean=0.0, std=1.0, 
                 act_fun=nn.ReLU,
                 sizes=None):
        super(Perceptron_Grads, self).__init__()
        self.mean = mean
        self.std = std
        self.act_fun = act_fun()
        self.sizes = sizes if sizes else [50, 2]
        self.linears = nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(len(self.sizes)-2)])
        self.fc_f = nn.Linear(self.sizes[-2], self.sizes[-1])  
        self.fc_grad = nn.Linear(self.sizes[-2], d*self.sizes[-1]) 

    def forward(self, x):
        x = x.view([x.shape[0], -1])
        x = (x - self.mean) / self.std
        
        for linear in self.linears:
            x = linear(x)
            x = self.act_fun(x)
        
        f_out = self.fc_f(x)
        grad_out = self.fc_grad(x)
        
        return f_out, grad_out


model_class=Perceptron_Grads



from util import flatten_parameters, get_param_properties,eval_losses, eval_model,norm_torch, compute_consensus_torch, normal_torch, effective_sample_size
from torch.func import vmap

N = 100
nodes = 200
models = [model_class(sizes=[d,nodes,1]) for _ in range(N)]
model = models[0]
criterion=nn.MSELoss()
pnames = [p[0] for p in model.named_parameters()]
w = flatten_parameters(models, pnames).to(device)
pprop = get_param_properties(models, pnames=pnames)


def custom_loss(x, y, grads, loss_fct, model, w, pprop,epsilon):
    with torch.no_grad():
        y_pred,grad_pred=eval_model(x, model, w, pprop)
        return loss_fct(y_pred, y)+epsilon*loss_fct(grad_pred, grads)
    
def custom_losses(x, y, grads, loss_fct, model, w, pprop,epsilon):
    return vmap(custom_loss, (None, None,None, None, None, 0, None,None))(x, y, grads, loss_fct, model, w, pprop,epsilon)




class objective:
    def __init__(self, train_loader, N, device, model, pprop):
        self.train_loader = train_loader
        self.data_iter = iter(train_loader)
        self.N = N
        self.epochs = 0
        self.device = device
        self.loss_fct = criterion
        self.model = model
        self.pprop = pprop
        self.set_batch()

    def __call__(self, w):
        return eval_losses(self.x, self.y, self.loss_fct, self.model, w[0,...], self.pprop)

    def set_batch(self,):
        (x,y) = next(self.data_iter, (None, None))
        if x is None:
            self.data_iter = iter(self.train_loader)
            (x,y) = next(self.data_iter)
            self.epochs += 1
            
        self.x = x.to(self.device)
        self.y = y.to(self.device)

class objective_grads:
    def __init__(self, train_loader, N, device, model, pprop,epsilon):
        self.train_loader = train_loader
        self.data_iter = iter(train_loader)
        self.N = N
        self.epochs = 0
        self.device = device
        self.loss_fct = criterion
        self.model = model
        self.pprop = pprop
        self.epsilon=epsilon
        self.set_batch()

    def __call__(self, w):
        return custom_losses(self.x, self.y, self.grads, self.loss_fct, self.model, w[0,...], self.pprop,self.epsilon)

    def set_batch(self,):
        (x,y,grads) = next(self.data_iter, (None, None,None))
        if x is None:
            self.data_iter = iter(self.train_loader)
            (x,y,grads) = next(self.data_iter)
            self.epochs += 1
            
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.grads = grads.to(self.device)


kwargs = {'alpha': 100.0,
        'dt': 0.1,
        'sigma': 0.1,
        'lamda': 1.0,
        'max_it': 200,
        'verbosity':0,
        'batch_args':{'batch_size':N},
        'check_f_dims':False}

epsilon = 10
f = objective_grads(train_loader, N, device, model, pprop,epsilon)
resampling =  rsmp.resampling([rsmp.loss_update_resampling(M=1, wait_thresh=40)], 1)
noise = anisotropic_noise(norm = norm_torch, sampler = normal_torch(device))

dyn = CBO(f, f_dim='3D', x=w[None,...], noise=noise,
          norm=norm_torch,
          copy=torch.clone,
          normal=normal_torch(device),
          compute_consensus=compute_consensus_torch,
          post_process = lambda dyn: resampling(dyn),
          **kwargs)
sched = effective_sample_size(maximum=1e7, name='alpha')


t1=time.time()
e = 1
mse_list=[]

while f.epochs < 10 :
    
    dyn.step()
    sched.update(dyn)
    f.set_batch()
    if e != f.epochs:
        e = f.epochs
        loss=0
        tot=0
        for x,y,g in iter(val_loader) :
                loss += custom_loss(x,y,g,criterion,model,dyn.best_particle[0,...],pprop,0)*x.size(0)
                tot += x.size(0)
        error=loss/tot
        mse_list.append(error)
        print(30*'-')
        print('Epoch: ' +str(f.epochs))
        print('Error :',error.item())
        print(30*'-')

t2=time.time()




plt.figure()
plt.plot([k for k in range(len(mse_list))],mse_list)
plt.title('Error for the CBO method')
plt.text(2, 550000, 'Dataset of '+str(n)+' points, ' +str(d) +' dimensions', fontsize=12)
plt.text(2, 540000, 'epsilon = '+str(epsilon)+' , one layer and '+str(nodes)+' nodes', fontsize=12)
plt.text(2, 530000, 'final loss = ' + str(mse_list[-1].item()), fontsize=12)
plt.show()

