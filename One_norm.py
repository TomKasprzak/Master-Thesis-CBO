

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




def f1(X):
    d = X.size(0)
    indices = torch.arange(1, d + 1, dtype=X.dtype, device=X.device)
    terms = d / 4 / (d / 4 + (X - (-1) ** indices / (indices + 1)) ** 2)
    product = torch.prod(terms)
    return product

def f2(x):
    d = x.size(0)
    half_d = d // 2
    prod1 = torch.prod(1 - x[:half_d] / (4 ** torch.arange(1, half_d + 1, dtype=torch.float32)))
    prod2 = torch.prod(torch.cos(16 * x[half_d:] / (2 ** torch.arange(half_d + 1, d + 1, dtype=torch.float32))))

    result = prod1 * prod2    
    return result

def f3(X) :
    d = X.size(0)
    alpha = torch.ones_like(X)
    x = torch.matmul(X.t(), alpha)/(2*d)
    return torch.exp(-x)

def f4(X):
    alpha = 10**(-1)*torch.ones_like(X)
    x = torch.matmul(X.t(), alpha)
    return (x - 5)**2 * (2 + torch.cos(x * 2 * torch.tensor(np.pi)))

def gradf(x,f):
    jacobian_fn = jacrev(f)
    grads = jacobian_fn(x)
    return(grads)

def hessianf(x,f) :
    hessian_fn = hessian(f)
    hessian = hessian_fn(x)
    return(hessian)




n = 50000
d = 500 # Dimensionality of R^d

# Generate random data
X = torch.rand(n, d,requires_grad = True)
Y = torch.tensor([f4(x) for x in X])
grads = torch.stack([gradf(x,f4) for x in X])

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
    



model_class=Perceptron




from util import flatten_parameters, get_param_properties,eval_losses, eval_model,norm_torch, compute_consensus_torch, normal_torch, effective_sample_size
from torch.func import vmap

N = 100
nodes = 100

models = [model_class(sizes=[d,nodes,1]) for _ in range(N)]
model = models[0]
criterion=nn.MSELoss()
pnames = [p[0] for p in model.named_parameters()]
w = flatten_parameters(models, pnames).to(device)
pprop = get_param_properties(models, pnames=pnames)


def custom_loss(x, y, grads, loss_fct, model, w, pprop,epsilon,kappa):
    with torch.no_grad():
        def model_output(x) :
            return(eval_model(x, model, w, pprop))
        
        def single_sample_model_output(x_single):
            return model_output(x_single.unsqueeze(0)).squeeze(0)

        y_pred = model_output(x)
        jacobian_fn = jacrev(single_sample_model_output)
        grads_pred = vmap(jacobian_fn)(x).squeeze(1)
        reg_term = torch.norm(w,p=1)
        return loss_fct(y_pred, y)+epsilon*loss_fct(grads_pred, grads)+kappa*reg_term # we have now also the one norm of the parameter to optimise
    
def custom_losses(x, y, grads, loss_fct, model, w, pprop,epsilon,kappa):
    return vmap(custom_loss, (None, None,None, None, None, 0, None,None,None))(x, y, grads, loss_fct, model, w, pprop,epsilon,kappa)



class objective_grads:
    def __init__(self, train_loader, N, device, model, pprop,epsilon,kappa):
        self.train_loader = train_loader
        self.data_iter = iter(train_loader)
        self.N = N
        self.epochs = 0
        self.device = device
        self.loss_fct = criterion
        self.model = model
        self.pprop = pprop
        self.epsilon = epsilon
        self.kappa = kappa
        self.set_batch()

    def __call__(self, w):
        return custom_losses(self.x, self.y, self.grads, self.loss_fct, self.model, w[0,...], self.pprop,self.epsilon,self.kappa)

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

epsilon = 1
kappa1 = 1e-2

f = objective_grads(train_loader, N, device, model, pprop,epsilon,kappa1)
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
e = -1
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
                loss += custom_loss(x,y,g,criterion,model,dyn.best_particle[0,...],pprop,0,0)*x.size(0)
                tot += x.size(0)
        error=loss/tot
        mse_list.append(error)
        print(30*'-')
        print('Epoch: ' +str(f.epochs))
        print('Error :',error.item())
        print(30*'-')

t2=time.time()

plt.figure()
plt.plot([k for k in range(len(mse_list))],mse_list, c ='r')
plt.title('Error for the CBO method '+'\n'+'Dataset of '+str(n)+' points'+' of dimension '+str(d)+'\n'+'epsilon = '+str(epsilon)+', kappa = '+str(kappa1)+' , one layer and '+str(nodes)+' nodes'+'\n'+'final loss = ' + str(mse_list[-1].item()))
plt.show()

w_norm = dyn.best_particle[0,...]



kwargs = {'alpha': 100.0,
        'dt': 0.1,
        'sigma': 0.1,
        'lamda': 1.0,
        'max_it': 200,
        'verbosity':0,
        'batch_args':{'batch_size':N},
        'check_f_dims':False}

epsilon = 1
kappa = 0

f = objective_grads(train_loader, N, device, model, pprop,epsilon,kappa)
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
e = -1
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
                loss += custom_loss(x,y,g,criterion,model,dyn.best_particle[0,...],pprop,0,0)*x.size(0)
                tot += x.size(0)
        error=loss/tot
        mse_list.append(error)
        print(30*'-')
        print('Epoch: ' +str(f.epochs))
        print('Error :',error.item())
        print(30*'-')

t2=time.time()

plt.figure()
plt.plot([k for k in range(len(mse_list))],mse_list, c ='r')
plt.title('Error for the CBO method '+'\n'+'Dataset of '+str(n)+' points'+' of dimension '+str(d)+'\n'+'epsilon = '+str(epsilon)+', kappa = '+str(kappa)+' , one layer and '+str(nodes)+' nodes'+'\n'+'final loss = ' + str(mse_list[-1].item()))
plt.show()

w_0 = dyn.best_particle[0,...]


# Let's try to quantify the robustness of this technic
# 


def add_noise(X, noise_factor=0.5):
    noise = noise_factor * torch.randn(X.size())
    return X + noise

def mask_features(X, mask_prob=0.1):
    mask = torch.bernoulli(torch.full(X.shape, mask_prob))
    return X * (1 - mask)

def add_outliers(X, outlier_factor=10):
    outliers = torch.bernoulli(torch.full(X.shape, 0.05))
    return X + outliers * outlier_factor * torch.randn(X.shape)


"""
We create small variation of the dataset
"""
X_test_noisy = add_noise(X_val)
X_test_masked = mask_features(X_val)
X_test_outliers = add_outliers(X_val)

X_noisy_pred = eval_model(X_test_noisy,model,w_norm,pprop)
X_masked_pred = eval_model(X_test_masked,model,w_norm,pprop)
X_outliers_pred = eval_model(X_test_outliers,model,w_norm,pprop)

loss_noisy = criterion(X_noisy_pred,y_val)
loss_masked = criterion(X_masked_pred,y_val)
loss_outliers = criterion(X_outliers_pred,y_val)





"""
We note the results for both test : with the values of kappa = 0 and  kappa = 0.01
"""
print('Kappa : '+str(kappa1))

print(30*"-")

print('The loss on the noisy dataset is : ' +str(loss_noisy.item()))
print(30*"-")

print('The loss on the masked dataset is : ' +str(loss_masked.item()))
print(30*"-")

print('The loss on the dataset with outliers is : ' +str(loss_outliers.item()))
print(30*"-")



X_noisy_pred = eval_model(X_test_noisy,model,w_0,pprop)
X_masked_pred = eval_model(X_test_masked,model,w_0,pprop)
X_outliers_pred = eval_model(X_test_outliers,model,w_0,pprop)

loss_noisy = criterion(X_noisy_pred,y_val)
loss_masked = criterion(X_masked_pred,y_val)
loss_outliers = criterion(X_outliers_pred,y_val)




print('Kappa : '+str(kappa))

print(30*"-")

print('The loss on the noisy dataset is : ' +str(loss_noisy.item()))
print(30*"-")

print('The loss on the masked dataset is : ' +str(loss_masked.item()))
print(30*"-")

print('The loss on the dataset with outliers is : ' +str(loss_outliers.item()))
print(30*"-")

