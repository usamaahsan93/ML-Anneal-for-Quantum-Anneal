from torch.autograd import Variable
import dwavebinarycsp as dbc

import numpy as np
import torch
from math import ceil
from OptimizerANDOptimizee import w
#from baseFunctions import discrete
#import ising_numba

#from ising_numba import dokify, getLowestResponse,ising_brute

#from functionLib import 
#import neal
#from primeFactor import MLOptim

#import dimod
#from OptimTrainer import w, MLOptim

#from OptimizeeClass import IsingData, IsingModel
#from Optimizer_LTO import Optimizer
#import pickle as pkl

def dict2Int_1(sample,d):
    i=0
    a_vars=[]
    
    while(True):
        p=d+str(i)
        if p in sample.keys():
            a_vars.append(p)
        else:
            break
        i+=1
#    print(a_vars)
    a = 0
    for lbl in reversed(a_vars):
        if sample[lbl] ==-1:
            a = (a << 1) | 0
#            print(sample[lbl])
        else:
            a = (a << 1) | 1
#            print(sample[lbl])
#    print('Digit Completed')
    return int(a)

def dict2Int(sample):
    return dict2Int_1(sample,'a'),dict2Int_1(sample,'b')


#def MLOptim(qBits,H=None,J=None,sigma=None,sweeps=100):
#    with open('3Q_100x100_HJNorm.optm','rb') as f:
#        quad_optimizer=pkl.load(f)
#    f.close()
#    
#    opt = w(Optimizer())
#    opt.load_state_dict(quad_optimizer)
#
#    energy,J,h,sigma=do_fit(opt, None, IsingData, IsingModel, 1, optim_it=sweeps, out_mul=1.0, qBits=qBits,should_train=False,h=H, j=J,sigma=None)
#    allSigma=[torch.Tensor.cpu(sigma[i]).detach().numpy() for i in range(len(sigma))]
#    return energy, h, J, allSigma


def sigma2int(sigma,hKeys):
    dictSample=dict(zip(hKeys,sigma))
#    print(dictSample)
    return dict2Int(dictSample)  


def dok2mat(h,J, convert2Tensor=False):
    _J=np.zeros([len(h),len(h)])
    hkeys,hvals = list(h.keys()),list(h.values())
    hIdx=dict(zip(hkeys,range(len(hkeys))))
    
    for i,j in J.keys():
        r=hIdx[i]
        c=hIdx[j]
        _J[r,c]=J[(i,j)]
    
    if convert2Tensor:
        h=w(Variable(torch.Tensor(hvals)))
        _J=w(Variable(torch.Tensor(_J)))   
        return h,_J
    
    else:
        h=np.array(hvals)
        return h,_J


def constructBQM(P):
    pBit=len(bin(P))-2 
    abBit=ceil(pBit/2)
    
    print('pBit : {}, abBit : {}'.format(pBit,abBit))

    csp = dbc.factories.multiplication_circuit(abBit)
    
    bqm = dbc.stitch(csp, min_classical_gap=.1)
#    return bqm,bqm

    p_vars=[]
    for i in range(pBit):
            p_vars.append('p'+str(i))
#        else:
#            break
        
    fixed_variables = dict(zip(reversed(p_vars), "{:b}".format(P)))
    fixed_variables = {var: int(x) for(var, x) in fixed_variables.items()}
    
    print(fixed_variables)
    for var, value in fixed_variables.items():
        bqm.fix_variable(var, value)
    
    bqm_ising=bqm.to_ising()
    h=bqm_ising[0]
    J=bqm_ising[1]      
    
    return h,J# , abBit


#    
#if __name__=='__main__':
#    P=23*17
#    hDict,JDict,abBit=constructBQM(P)
#    h,J=dok2mat(hDict,JDict,convert2Tensor=True)
#    
##    hNumpy=h
##    JNumpy=J
#    
#    hNumpy=h.cpu().numpy()
#    JNumpy=J.cpu().numpy()   
#    nn=np.linalg.norm(hNumpy)*np.linalg.norm(JNumpy)
#
#    r = neal.SimulatedAnnealingSampler().sample_ising(dokify(hNumpy), dokify(JNumpy),num_reads=100)    
#    rn = neal.SimulatedAnnealingSampler().sample_ising(dokify(hNumpy/nn), dokify(JNumpy/nn),num_reads=100)
#    
#    s,e=getLowestResponse(r)
#
#    sn,en=getLowestResponse(rn)
#    print("Unnormalized energy:",ising_brute(hNumpy,JNumpy,s)[0],ising_brute(hNumpy,JNumpy,sn)[0])
#    print("Normalized energy:",ising_brute(hNumpy/nn,JNumpy/nn,s)[0],ising_brute(hNumpy/nn,JNumpy/nn,sn)[0])
#            
#
#    hDictnn=dict(zip(hDict.keys(),list(hDict.values())/nn))
#    JDictnn=dict(zip(JDict.keys(),list(JDict.values())/nn))
#    
#    print('Testing Direct')
#    for i in s:
#        k=dict(zip(hDict.keys(),i))       
#        print('Energy : ',dimod.ising_energy(k,hDict,JDict),' Numbers : ',dict2Int(k,abBit))
#
#    print('Testing Normalized')
#    for i in sn:
#        k=dict(zip(hDict.keys(),i))       
#        print('Energy : ',dimod.ising_energy(k,hDictnn,JDictnn),' Numbers : ',dict2Int(k,abBit))
#     
#    k=MLOptim(qBits=len(h),H=h/nn,J=J/nn)
#    allSigma=k[1]
#    
#    sigma=[]
#    for i in allSigma:
#        p=discrete(i)
#        sigma.append(p)
# 








       
#for i in range(100):
#    h,_,_=constructBQM(i+2)
#    h=len(h)    sampler.parameters['sweeps']=sweep
    

#    sampler.parameters['num_reads']=numRead
#    response = sampler.sample_ising(h, J,sweeps=sweep,num_reads=numRead)
#    energies=response.record['energy']
#    sigmas=response.record['sample']
#    
    
    
#    with open('Qbits.txt','a') as f:
#        f.write('Value : {} , Qubits : {}\n'.format(i+2,h)
#        f.close()

       
#    for i in range(len(sigma)):
#        for j in range(len(sn)):
#            klm=np.all(sn[j] == sigma[i])
#            print(klm)
#            if klm:
#                print(i,j)
#                break;
#                
                
                
                
#    print('ML Output Normalized')  
#    for i in sigma:
#        k=dict(zip(hDict.keys(),i))       
#        print('Energy : ',dimod.ising_energy(k,hDictnn,JDictnn),' Numbers : ',dict2Int(k,abBit))
        
#    print(k[0])
    
    
#    print('Now Checking Int SA normalized values')   
#    for i in sn:    
#        k=dict(zip(hDict.keys(),i))
#        print(dict2Int(k,abBit))    
#        
#        
#    print('Now Checking Int SA values')   
#    for i in s:    
#        k=dict(zip(hDict.keys(),i))
#        print(dict2Int(k,abBit))  

#    print(sn)
    
    
    
#    #RECHECKING
#    import dimod
#    
##    for i in s:
##        print(dimod.ising_energy(i,dokify(hNumpy),dokify(JNumpy)))
##    
##    for i in sn:
##        print(dimod.ising_energy(i,dokify(hNumpy),dokify(JNumpy)))
#    
#
##    p=MLOptim(qBits=len(h),H=h,J=J)
#    pn=MLOptim(qBits=len(h),H=h/nn,J=J/nn)
#    print('Checking ML')
#    for i in pn[-1]:
#        k=discrete(i)
#        k=dict(zip(hDict.keys(),k))
#        print(dict2Int(k,abBit))        
#    
#    1/0
##    sp=[]
##    for i in p[3]:
##        k=discrete(i)
##        k=np.array(k)
##        k[k<=0]=0
##        sp.append(k)
#        
#    spn=[]
#    for i in pn[3]:
#        k=discrete(i)
#        k=np.array(k)
#        k[k<=0]=0
#        spn.append(k)
#        
##    print('Checking Int at all values')
##    for i in sp:
##        k=dict(zip(hDict.keys(),i))
##        print(dict2Int(k,abBit))
#    
#    print('Checking Int at all normalized values')   
#    for i in spn:
#        k=dict(zip(hDict.keys(),i))
##        print(dict2Int(k,abBit))
#    
#    print('Checking Int SA values')   
#    for i in s:    
#        k=dict(zip(hDict.keys(),i))
#        print(dict2Int(k,abBit))
#    
#    print('Checking Int SA normalized values')   
#    for i in sn:    
#        k=dict(zip(hDict.keys(),i))
#        print(dict2Int(k,abBit))
#    