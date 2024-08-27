

import cbx as cbx
from cbx.dynamics.cbo import CBO

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD



from cbx.noise import anisotropic_noise
import cbx.utils.resampling as rsmp

import time



device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,)) 
])

#size 28x28
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


"""
Let's define our Neural network using Pytorch
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
        self.sizes = sizes if sizes else [784, 10]
        self.linears = nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(len(self.sizes)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.sizes[i+1], track_running_stats=False) for i in range(len(self.sizes)-1)])
        self.sm = nn.Softmax(dim=1)

    def __call__(self, x):
        x = x.view([x.shape[0], -1])
        x = (x - self.mean)/self.std
        
        for linear, bn in zip(self.linears, self.bns):
            x = linear(x)
            x = self.act_fun(x)
            x = bn(x)

        # apply softmax
        x = self.sm(x)
        return x

model_class=Perceptron


from util import flatten_parameters, get_param_properties, eval_losses, norm_torch, compute_consensus_torch, normal_torch, eval_acc,effective_sample_size
N = 10
models = [model_class(sizes=[784,10]) for _ in range(N)]
model = models[0]
pnames = [p[0] for p in model.named_parameters()]
w = flatten_parameters(models, pnames).to(device)
pprop = get_param_properties(models, pnames=pnames)


"""
The objective function
"""

class objective:
    def __init__(self, train_loader, N, device, model, pprop):
        self.train_loader = train_loader
        self.data_iter = iter(train_loader)
        self.N = N
        self.epochs = 0
        self.device = device
        self.loss_fct = nn.CrossEntropyLoss()
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


"""
Let's use the CBX library to find the parameters of the Neural Network
"""


kwargs = {'alpha': 50.0,
        'dt': 0.1,
        'sigma': np.sqrt(0.1),
        'lamda': 1.0,
        'max_it': 200,
        'verbosity':0,
        'batch_args':{'batch_size':N},
        'check_f_dims':False}

f = objective(train_loader, N, device, model, pprop)
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
e = 0
accuracy_list=[]

while f.epochs < 30:
    
    dyn.step()
    sched.update(dyn)
    f.set_batch()
    if e != f.epochs:
        e = f.epochs
        print(30*'-')
        print('Epoch: ' +str(f.epochs))
        acc = eval_acc(model, dyn.best_particle[0,...], pprop, test_loader)
        accuracy_list.append(acc.item())
        print('Accuracy: ' + str(acc.item()))
        print(30*'-')
t2=time.time()




plt.figure()
plt.plot([k for k in range(len(accuracy_list))],accuracy_list , c='r')
plt.title('Accuracy for the CBO method : zero hidden layer, '+str(N)+' particles')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.show()


# SGD


"""
We are going to use SGD now to find the parameters of the neural network

"""


model = Perceptron(sizes=[784,10])
optimizer = SGD(model.parameters(), lr = 0.1)
criterion = nn.CrossEntropyLoss()


"""
We compute one step of training for our model
"""
def train_epoch(trainloader, net, optimizer, criterion):
    net.train()  # Set the model to training mode
    epoch_loss = 0.0 #set the loss to 0
    total_points = 0 #set the total points to 0
    accuracy = 0
    for inputs, labels in trainloader:
        optimizer.zero_grad()  # Zero the gradients

        # we apply the forward method
        outputs = net(inputs)
        
        # we compute the loss
        loss = criterion(outputs, labels)
        
        #we compute the error over the prediction
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # we update the total loss and the total points for the epoch
        epoch_loss += loss.item() * inputs.size(0)
        total_points += inputs.size(0)

    return epoch_loss/total_points,accuracy/total_points #we return the average 

def test_epoch(testloader, net, criterion) :
    net.eval()
    epoch_loss = 0
    accuracy = 0
    total_points = 0
    for inputs, labels in testloader:

        # we apply the forward pass
        outputs = net(inputs)
        
        # we compute the loss
        loss = criterion(outputs, labels)

        # we update the total loss for the epoch
        epoch_loss += loss.item() * inputs.size(0)

        #we compute the error over the prediction
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()

        total_points += inputs.size(0)
        
    average_loss = epoch_loss/total_points
    error = accuracy/total_points

    return(average_loss,error) #we return the average


def model_performance(model,optimizer,criterion,n_epoch,train_loader,test_loader,history=False) :
    train_loss_list=[]
    test_loss_list=[]
    test_accuracy_list=[]
    for i in range(n_epoch) :
        train_loss,train_accuracy = train_epoch(train_loader, model, optimizer, criterion)
        test_loss,test_accuracy = test_epoch(test_loader, model, criterion)
        print(f"Epoch: {i} | Train Loss: {train_loss} | Test Loss: {test_loss} | Test accuracy: {test_accuracy}")
        test_accuracy_list.append(test_accuracy)
        if history : 
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
    return(test_accuracy_list)
    
n_epoch=30

t1=time.time()
test_accuracy_list=model_performance(model,optimizer,criterion,n_epoch,train_loader,test_loader,history=False)
t2=time.time()




plt.figure()
plt.plot([n_epoch for n_epoch in range(len(test_accuracy_list))],test_accuracy_list,c='r')
plt.xlabel('n_epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy of the model with SGD')
plt.show()

