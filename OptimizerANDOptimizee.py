import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
#from numpy import eye

USE_CUDA = True


qBits=18

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

class IsingData:
    def __init__(self,h=None,j=None): 
        self.sgma=[]
#        
#        if 'J' in kwargs.keys() and 'H' in kwargs.keys():
#            
#            h=kwargs['H']
#            j=kwargs['J']
#            
        if h is None or j is None:
#            print('Generating h and J of Ising Model')
            self.H,self.J=self.generate_data()
            
        else:
#            print('Using given h and J of Ising Model')
            self.H=h
            self.J=j
#        else:
#            self.H,self.J=self.generate_data()
    
    def get_loss(self, sigma):
#        print('*'*20+'\n')
#        print(sigma)
        
        self.sgma.append(sigma)
        
        sigmaNorm=sigma.norm()
        sigma=sigma/(sigmaNorm+1e-10)
        
#        print(sigma)
#        print('*'*20+'\n')
        
        K=sigma.matmul(self.J.matmul(sigma)) + 1*self.H.matmul(sigma)        
        
        return K
    
    def generate_data(self):
        
#        J=torch.randn(qBits,qBits)
        J=torch.randn((qBits,qBits))
        #Making Diagonal Empty
        J=J*(1-torch.eye(qBits))
        J = w(Variable(J))
        
#        H = w(Variable(torch.randn(qBits)))
        H = w(Variable(torch.randn(qBits)))
#        J = w(Variable(torch.ones(qBits,qBits)))
#        H = w(Variable(torch.ones(qBits)))
        
#        diagZero=abs((eye(qBits) - 1) /-1)
#        diagZeroTensor=w(Variable(torch.tensor(diagZero,dtype=torch.float32)))
#        J=J.mul(diagZeroTensor)
######################################################        
#        Jt=J.transpose(0,1)       
#        J=(J+Jt)/2
        J=torch.triu(J)
##########################################################
        norms=H.norm()*J.norm()
        J=J/norms
        H=H/norms
        
#        J=J/H.norm()*J.norm()
#        H=H/H.norm()*J.norm()
        

        return H,J
        

#################################################################################   
class IsingModel(nn.Module):
    def __init__(self, sigma=None):
        super().__init__()
        # Note: assuming the same optimization for sigma as for
        # the function to find out itself.

        if sigma is None:
            self.sigma = nn.Parameter(torch.randn(qBits))
            
        else:
            self.sigma = sigma

    def forward(self, target):
        return target.get_loss(self.sigma)
    
    def all_named_parameters(self):

        return [('sigma', self.sigma)]
    
    
    
#########################################################################################
class Optimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0): #hidden_sz=20
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        
    def forward(self, inp, hidden, cell):
        if self.preproc:
            # Implement preproc described in Appendix A
            
            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = torch.abs(inp) >= self.preproc_threshold
            inp2[:, 0][keep_grads] = torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads])
            
            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = float(np.exp(self.preproc_factor)) * inp[~keep_grads]
            inp = w(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)
    