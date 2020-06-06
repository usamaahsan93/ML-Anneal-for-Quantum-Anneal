import itertools
import numpy as np
import dimod
from scipy import sparse
#from OptimizeeClass import 
import torch
import neal
#from dwave.system.samplers import DWaveSampler
#from dwave.system.composites import EmbeddingComposite
import matplotlib.pyplot as plt
import pickle as pkl
from OptimTrainer import do_fit
from OptimizerANDOptimizee import Optimizer,w,IsingData, IsingModel
#import torch.nn as nn
#from torch.autograd import Variable

#
#USE_CUDA = True
#
#def w(v):
#    if USE_CUDA:
#        return v.cuda()
#    return v

def convertCompatible(h,J):
    h=dict(zip(range(len(h)),h))
    J=sparse.dok_matrix(J)
    J=dict(zip(J.keys(),J.values()))
    return h,J

def ising(sigma,h,J):
    h,J=convertCompatible(h,J)
    e=dimod.ising_energy(sigma,h,J)
    return e

def simulatedNeal(h,J,i,j):
    h,J=convertCompatible(h,J)
#    print(h)
#    print(J)
    sampler=neal.SimulatedAnnealingSampler()
    sampler.parameters['sweeps']=i
    sampler.parameters['num_reads']=j

    response = sampler.sample_ising(h, J,sweeps=i,num_reads=j)
#    k=np.array(response.record)
    energies=response.record['energy']
    sigmas=response.record['sample']
#    k=np.min(energies)
############################    
#####    minIdx=list(energies).index(np.min(energies))
 ##########################   
#####    minSigma=sigmas[minIdx]
#####    minSigma=minSigma.astype('float32')
#    p=[k[i][0] for i in range(len(k))]
#    print(response.record)
#    for datum in response.data(['sample', 'energy']):
#        energy0.append(datum.energy)
#        sigma=list(datum.sample.values())
    
#    print(response.record,min(energy0))
#    return sigma,min(energy0)
    
#####    minSigma=nn.Parameter(torch.from_numpy(minSigma))
    return sigmas,energies


def discrete(q):
    q=np.array(q)
    q[q>0]=1
    q[q<=0]=-1
    q=[int(i) for i in q]
    return list(q)

def bruteForce(h,J):
    r=len(h)
    z = itertools.product([-1, 1], repeat=r)
    if r>15:
        print('WARNING: Going brute force on large data will take time')
        
    l=[]

    idx=0
    for i in z:
        sigma=list(i)
        loss=ising(sigma,h,J)
#        print(idx,'   ',loss)
        l.append(loss)
        idx+=1
    return l

def MLOptim(H=None,J=None,sigma=None,sweeps=1):
    opt='S_5Q_epoch100outOf1000.optm'
    print('Running Optimizer : '+opt)
    with open(opt,'rb') as f:
        quad_optimizer=pkl.load(f)
    f.close()
    
    opt = w(Optimizer())
    opt.load_state_dict(quad_optimizer)
#    np.random.seed(0)  
    
    
    data=do_fit(opt, None, IsingData, IsingModel, 1, optim_it=sweeps, out_mul=1.0, should_train=False,isingLink=J, isingBias=H,sigma=None)
    
#    energy=data['allLoss']
#    J=data['J']
#    h=data['h']
    sigma=data['sigma']
    
    allSigma=[torch.Tensor.cpu(sigma[i]).detach().numpy() for i in range(len(sigma))]
#    sigma=list(torch.Tensor.cpu(sigma[0]).detach().numpy())
    
    result={'allSigma':allSigma,
            'J':data['J'],
            'h':data['h']
            }
#    return sigma, energy, h, J, allSigma
    return result
####################################################################################
if __name__=='__main__':
    
    print('Generating H and J and Calculating')
#    mlSigma,mlEnergy,hTensor,JTensor,allSigma=MLOptim()
    output=MLOptim()
    JTensor=output['J']
    hTensor=output['h']
#    print(JTensor[0])
#    hTensor=w(Variable(torch.randn(23)))
#    J=w(Variable(torch.randn(23,23)))
#    diagZero=abs((eye(qBits) - 1) /-1)
#    diagZeroTensor=w(Variable(torch.tensor(diagZero,dtype=torch.float32)))
#    J=J.mul(diagZeroTensor)
#    Jt=J.transpose(0,1)
#    
#    J=(J+Jt)/2
#    J=torch.triu(J)
#        
#    
#    
#    JTensor=torch.triu(JTensor)
    ##########################################################
    ##########################################################
    ##########################################################
    ##########################################################

    J=torch.Tensor.cpu(JTensor).detach().numpy()
    h=torch.Tensor.cpu(hTensor).detach().numpy()
#    print(J[0])
    
#    J=np.ones(J.shape)-np.eye(J.shape[1])
#    J=np.triu(J)
#    h=np.ones(h.shape)
#    
#    norms=np.linalg.norm(J)*np.linalg.norm(h)
#    J=J/norms
#    h=h/norms
    
    ##########################################################
#    JTensor=w(torch.Tensor(J))
#    hTensor=w(torch.Tensor(h))
    ##########################################################
    if len(h)>=15:
        useBruteforce=False
    else:
        useBruteforce=True
    
    if useBruteforce:
        print('Bruteforce Running')
        bfEnergy=min(bruteForce(h,J))
        print('Bruteforce Energy : ',bfEnergy)
    
    else:
        bfEnergy=np.nan
    #mlSigma=discrete(mlSigma)
    #mlEnergy0=ising(mlSigma,h,J)
    num_reads=100
    sweeps=100
    
    print('Processing SA')
    allSAEnergy=[]
    for i in np.arange(1,sweeps+1,1):
        print('Running Neal for Sweeps : ',i)
    #    for j in np.arange(1,10,1):
        _,energy=simulatedNeal(h,J,i,num_reads)
        allSAEnergy.append(energy)
    
    
    
#    normalized=hTensor.norm()*JTensor.norm()
##    normalized=1
#    JTensor=JTensor/normalized
#    hTensor=hTensor/normalized
#    print(JTensor[0])
#    1/0
    #allMLSigma=[discrete(allSigma[i]) for i in range(len(allSigma))]
    #allMLEnergy=[ising(allMLSigma[i],h,J) for i in range(len(allSigma))]
    #e.append(min(allMLEnergy))
    
#    Jt=torch.Tensor.cpu(JTensor).detach().numpy()
#    ht=torch.Tensor.cpu(hTensor).detach().numpy()
    
    print('Processing ML')
    e=[]
    for j in range(num_reads):       
#        mlSigma,mlEnergy,H_,J_,allSigma=MLOptim(H=hTensor,J=JTensor,sweeps=sweeps)#,sigma=sigma)
        mlOutput=MLOptim(H=hTensor,J=JTensor,sweeps=sweeps)
        allSigma=mlOutput['allSigma']
        
        print('Running for num_reads : ',j)
        allMLSigma=[discrete(allSigma[i]) for i in range(len(allSigma))]
        allMLEnergy=[ising(allMLSigma[i],h,J) for i in range(len(allSigma))]    
         
        e.append(allMLEnergy)
    
    
    medianMLline=np.min(np.array(e),axis=0)
    medianSAline=np.min(allSAEnergy,axis=1)
    
    
    #print(startSigma)
    #print(allMLSigma)
    
    
    ############################################################
    #minMLline=np.min(np.array(e),axis=0)
    #minSAline=np.min(allSAEnergy,axis=1)
    #
    #
    #
    #
    #
    #
    plt.close('all')
    plt.figure()
    
    plt.boxplot(np.array(e))
    plt.plot(range(1,sweeps+1),medianMLline,label='ML Output')    
    plt.plot((1,sweeps+1),[bfEnergy,bfEnergy],'g',label='Bruteforce Output')
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.boxplot(allSAEnergy)
    plt.plot(range(1,sweeps+1),medianSAline,'k--',label='Neal Output')
    plt.plot((1,sweeps+1),[bfEnergy,bfEnergy],'g',label='Bruteforce Output')
    #plt.plot([0,range(1,11)],[bfEnergy,bfEnergy],label='Bruteforce Output')
    plt.legend()
    plt.grid()
    
    
    plt.figure()
    plt.plot(range(1,sweeps+1),medianMLline,'o-',label='ML Output')   
#    plt.plot(range(1,sweeps+1),medianSAline,'k*--',label='Neal Output')
    plt.plot((1,sweeps+1),[bfEnergy,bfEnergy],'g',label='Bruteforce Output')
    plt.legend()
    plt.grid()
    #
    ########################################################
    plt.figure()
    plt.plot(range(1,sweeps+1),medianMLline,'o-',label='min ML Output')   
    plt.plot(range(1,sweeps+1),medianSAline,'ko--',label='min Neal Output')
    plt.plot((1,sweeps+1),[bfEnergy,bfEnergy],'g',label='Bruteforce Output')
    plt.legend()
    plt.grid()
#
#    print('Brute Force Direct : ',bfEnergy)
#    Jt=torch.Tensor.cpu(JTensor).detach().numpy()
#    ht=torch.Tensor.cpu(hTensor).detach().numpy()
#    klm=min(bruteForce(ht,Jt))
#    print('Brute Force Normalized : ',klm)
#    
#    plt.figure()
#    plt.plot(range(1,sweeps+1),medianMLline,'o-',label='min ML Output')   
##    plt.plot(range(1,sweeps+1),medianSAline,'ko--',label='min Neal Output')
#    plt.plot((1,sweeps+1),[klm,klm],'g',label='Bruteforce Output')
#    plt.legend()
#    plt.grid()