import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import tqdm
import copy
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("white")

#from OptimizeeClass import 
from OptimizerANDOptimizee import Optimizer,IsingData, IsingModel
import pickle as pkl

USE_CUDA = True

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def do_fit(OptimizerModel, OptimizerOptim, OptimizeeData, OptimizeeClass, unroll, optim_it, out_mul, isingBias=None,isingLink=None,should_train=True,sigma=None):
    if should_train:
        OptimizerModel.train()
    else:
        OptimizerModel.eval()
        unroll = 1
    
    target = OptimizeeData(h=isingBias,j=isingLink)
    optimizee = w(OptimizeeClass(sigma=sigma))
    n_params = 0
    for p in optimizee.parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [w(Variable(torch.zeros(n_params, OptimizerModel.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, OptimizerModel.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        OptimizerOptim.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)
                    
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, OptimizerModel.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, OptimizerModel.hidden_sz))) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            updates, new_hidden, new_cell = OptimizerModel(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()
            
        if iteration % unroll == 0:
            if should_train:
                OptimizerOptim.zero_grad()
                all_losses.backward()
                OptimizerOptim.step()
                
            all_losses = None
                        
            optimizee = w(OptimizeeClass(**{k: detach_var(v) for k, v in result_params.items()}))
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            
        else:
            optimizee = w(OptimizeeClass(**result_params))
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2
    
#    print(all_losses_ever)
    if should_train:        
        return all_losses_ever
    else:
        
        data={'allLoss':all_losses_ever,
              'J':target.J,
              'h':target.H,
              'sigma':target.sgma}
        
        return data
#        return [all_losses_ever, target.J, target.H, target.sgma]#optimizee.all_named_parameters()]

def fit_optimizer(OptimizeeData, OptimizeeClass, preproc=False, unroll=20, optim_it=100, n_epochs=20, n_tests=100, lr=0.001, out_mul=1.0):
    OptimizerModel = w(Optimizer(preproc=preproc))
    OptimizerOptim = optim.Adam(OptimizerModel.parameters(), lr=lr)
    
    best_net = None
    best_loss = np.inf
    
    for jkl in tqdm.trange(n_epochs):#), 'epochs'): # Initial n_epochs = 100
        for _ in range(1000):#, 'iterations'): # Initial 20
            do_fit(OptimizerModel, OptimizerOptim, OptimizeeData, OptimizeeClass, unroll, optim_it, out_mul, should_train=True)
        
        loss = (np.mean([np.sum(
                do_fit(OptimizerModel, OptimizerOptim, OptimizeeData, OptimizeeClass, unroll, optim_it, out_mul, should_train=False)['allLoss'])
        for _ in range(n_tests)]))
    
        print(loss)
        if loss < best_loss:
            print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(OptimizerModel.state_dict())
            with open('5Q_epoch'+str(jkl)+'outOf'+str(n_epochs)+'.optm','wb') as f:
                pkl.dump(best_net,f)
            
    return best_loss, best_net

if __name__=='__main__':
#    kwargs={'train':True}
    print('RUNNING ON {} QUBITS'.format(5))
    loss, quad_optimizer = fit_optimizer(IsingData, IsingModel, lr=0.01, n_epochs=1000) #lr=0.01 lr=0.03
#
##
#    with open('100Q_iter1000_epoch100.optm','wb') as f:
#        pkl.dump(quad_optimizer,f)

    print('DONE... at final loss : {}'.format(loss))
