import itertools
import numpy as np
import dimod
from scipy import sparse
#from OptimizeeClass import 
import torch
import neal
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import matplotlib.pyplot as plt
import pickle as pkl
from OptimTrainer import do_fit
from OptimizerANDOptimizee import Optimizer,IsingData, IsingModel,w
from SAVsMLConvergence import MLOptim
plt.close('all')
#USE_CUDA = True
#
#def w(v):
#    if USE_CUDA:
#        return v.cuda()
#    return v


def discrete(q):
    q=np.array(q)
    q[q>0]=1
    q[q<=0]=-1
    q=[int(i) for i in q]
    return list(q)

def convertCompatible(h,J):
    h=dict(zip(range(len(h)),h))
    J=sparse.dok_matrix(J)
    J=dict(zip(J.keys(),J.values()))
    return h,J

def ising(sigma,h,J):
    h,J=convertCompatible(h,J)
    e=dimod.ising_energy(sigma,h,J)
    return e

def simulatedNeal(h,J):
    h,J=convertCompatible(h,J)
    sampler=neal.SimulatedAnnealingSampler()
    response = sampler.sample_ising(h, J)
    
    for datum in response.data(['sample', 'energy']):
        energy=datum.energy
        sigma=list(datum.sample.values())
    return sigma,energy,response

def quantumOptim(h,J):
    h,J=convertCompatible(h,J)
    sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi', token='[EnterTokenHere]', solver='DW_2000Q_2_1'))
    response = sampler.sample_ising(h,J,num_reads=1000)

    Q_energy=[]
    Q_sigma=[]

    for datum in response.data(['sample', 'energy']):
        Q_energy.append(datum.energy)
        Q_sigma.append(list(datum.sample.values()))
      
    return Q_sigma[0],Q_energy[0]

#def MLOptim():
#    with open('S_20Q_epoch11_outof_1000_iter1000.optm','rb') as f:
#        quad_optimizer=pkl.load(f)
#    f.close()
#    
#    opt = w(Optimizer())
#    opt.load_state_dict(quad_optimizer)
#    np.random.seed(0)  
#
#    energy,J,h,sigma=do_fit(opt, None, IsingData, IsingModel, 1, 100,out_mul=1.0, should_train=False)
##    print(sigma)
#    sigma=list(torch.Tensor.cpu(sigma[-1]).detach().numpy())
#    return sigma, energy, h, J


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


def ploting(x,y,xLabel,yLabel):
    plt.figure()
    plt.plot([min(x)-0.5,0],[min(x)-0.5,0],'k')
    plt.scatter(x,y)
    plt.grid()
    plt.xlabel(xLabel,fontsize=16)
    plt.ylabel(yLabel,fontsize=16)
##########################################################################    

bfEnergyplot=[]
qEnergyplot=[]
saEnergyplot=[]
mlEnergyplot=[]



useQuantum=True
useBruteforce=False



for sameHJ in range(50):
    print('\n\nIteration No :',sameHJ)
    
    print('Solving for ML at same H and J')
    output=MLOptim(sweeps=50)
    JTensor=output['J']
    hTensor=output['h']
#    print(h)
    J=torch.Tensor.cpu(JTensor).detach().numpy()
    h=torch.Tensor.cpu(hTensor).detach().numpy()
    _sigma=[]
    _energy=[]
    
    mlSigma=output['allSigma']
    for i in mlSigma:
        k=discrete(i)
        _sigma.append(k)
        _energy.append(ising(k,h,J))
    
    e=min(_energy)
    s=_sigma[np.argmin(np.array(e))]
    
#    print(_energy)
    
    if useBruteforce:
        print('Solving for Bruteforce at same H and J')
        l=bruteForce(h,J)
        bfEnergy=min(l)
        bfSigma=np.argmin(l)
#       bfSigma=discrete(np.array(list(bin(bfSigma)[2:]),dtype=np.int))
        bfSigma=discrete(np.array(list(bin(bfSigma)[2:].zfill(len(h))),dtype=np.int))
    
    if useQuantum:
        print('Solving for Quantum at same H and J')
        qSigma,qEnergy=quantumOptim(h,J)
    
    print('Solving for Simulated Annealing for Same H and J')
    saSigma,saEnergy,r=simulatedNeal(h,J)
    
#    bfEnergy=min(l)
#    bfSigma=np.argmin(l)
##    bfSigma=discrete(np.array(list(bin(bfSigma)[2:]),dtype=np.int))
#    bfSigma=discrete(np.array(list(bin(bfSigma)[2:].zfill(len(h))),dtype=np.int))
    
    with open ('./Data_QP_Vs_ML/hJ_{}.txt'.format(sameHJ),'w+') as f:
        if useQuantum:
            f.write('[Quantum]\tSigma : '+str(np.array(qSigma))+'\tEnergy : '+str(np.array(qEnergy))+'\n\n')
        if useBruteforce:
            f.write('[Bruteforce]\tSigma : '+str(bfSigma)+'\tEnergy :'+str(bfEnergy)+'\n\n')
        
        f.write('[Anneal]\tSigma : '+str(np.array(saSigma))+'\tEnergy : '+str(np.array(saEnergy))+'\n\n')
        f.write('[ML]\t\tSigma : '+str(s)+'\tEnergy :'+str(e)+'\n\n\n')
        f.write('h : '+str(np.array(h))+'\n\n\nJ : '+str(np.array(J))+'\n\n')
        if useBruteforce:
            f.write('All Values of Bruteforce: '+str(l))
    

    saEnergyplot.append(saEnergy)
    mlEnergyplot.append(e)
    if useQuantum:
        qEnergyplot.append(qEnergy)
    if useBruteforce:
        bfEnergyplot.append(bfEnergy)
        
#############################################################    

    print('ML Sigma={}\tEnergy={}'.format(s,e))    
    print('SA Sigma={}\tEnergy={}'.format(saSigma,saEnergy))
    if useQuantum:
        print('Q  Sigma={} Energy={}'.format(qSigma,qEnergy))
        
    if useBruteforce:
        print('BF Sigma={}\tEnergy={}'.format(bfSigma,bfEnergy))
        
if useBruteforce:
    ploting(bfEnergyplot,mlEnergyplot,'Bruteforce Energy','ML Energy')
    ploting(bfEnergyplot,saEnergyplot,'Bruteforce Energy','Simulated Annealing Energy')

if useQuantum:
    ploting(qEnergyplot,mlEnergyplot,'Quantum Energy','ML Energy')
    ploting(qEnergyplot,saEnergyplot,'Quantum Energy','Simulated Annealing Energy')
    if useBruteforce:
        ploting(bfEnergyplot,qEnergyplot,'Bruteforce Energy','Quantum Energy')
        
ploting(saEnergyplot,mlEnergyplot,'Simulated Annealing Energy','ML Energy')

with open('graphPoints_ml_qp_sa.pkl','wb') as f:
    pkl.dump(mlEnergyplot,f)
    pkl.dump(qEnergyplot,f)
    pkl.dump(saEnergyplot,f)
