#CODE FROM https://docs.ocean.dwavesys.com/en/latest/examples/map_coloring.html
import dwavebinarycsp
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import matplotlib.pyplot as plt

import torch
from SAVsMLConvergence import MLOptim
from factorFunctions import dok2mat
import dimod
from scipy import sparse

import neal
import numpy as np
plt.close('all')
# Represent the map as the nodes and edges of a graph
provinces = ['A','B','C','D','E','F']
#neighbors = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'A')]
#neighbors = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('D', 'E'), ('D', 'F')]
#neighbors = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('D', 'E'), ('D', 'F'), ('E', 'A')]
#neighbors = [('A', 'B'), ('A', 'E'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('C', 'F'), ('D', 'A'), ('B', 'E')]
#neighbors = [('A', 'B'), ('A', 'C'), ('A', 'F'), ('B', 'D'), ('B', 'F'), ('D', 'F'), ('D', 'E'), ('F', 'E')]
#neighbors = [('A', 'C'), ('A', 'D'), ('B', 'C'), ('C', 'E'), ('E', 'D'), ('D', 'F')]
#neighbors = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E'), ('D', 'F'), ('E', 'F'), ('F', 'A')] # X
neighbors = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('B', 'E'), ('F', 'E')]#, ('D', 'F'), ('E', 'F'), ('F', 'A')]
# Function for the constraint that two nodes with a shared edge not both select
# one color
def not_both_1(v, u):
    return not (v and u)

# Valid configurations for the constraint that each node select a single color
one_color_configurations = {(0, 0, 1), (0, 1, 0), (1, 0, 0)}
colors = len(one_color_configurations)

# Create a binary constraint satisfaction problem
csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

# Add constraint that each node (province) select a single color
for province in provinces:
    variables = [province+str(i) for i in range(colors)]
    csp.add_constraint(one_color_configurations, variables)

# Add constraint that each pair of nodes with a shared edge not both select one color
for neighbor in neighbors:
    v, u = neighbor
    for i in range(colors):
        variables = [v+str(i), u+str(i)]
        csp.add_constraint(not_both_1, variables)

bqm = dwavebinarycsp.stitch(csp)

sampler=neal.SimulatedAnnealingSampler()
response = sampler.sample(bqm, num_reads=50)


##################################################################
#num_reads=250
#sweeps=10
#    
#print('Processing SA')
#allSAEnergy=[]
#for i in np.arange(1,sweeps+1,1):
#    print('Running Neal for Sweeps : ',i)
##    for j in np.arange(1,10,1):
#    response=sampler.sample(bqm, num_reads=num_reads,sweeps=i)
#    allSAEnergy.append(response.record['energy'])
##################################################################



# Function that plots a returned sample
def plot_map(sample):
    G = nx.Graph()
    G.add_nodes_from(provinces)
    G.add_edges_from(neighbors)
    # Translate from binary to integer color representation
    color_map = {}
    for province in provinces:
          for i in range(colors):
            print(province+str(i),'\t',sample[province+str(i)])
            if sample[province+str(i)]:
                color_map[province] = i
#                break
    # Plot the sample with color-coded nodes
    node_colors = [color_map.get(node) for node in G.nodes()]
    nx.draw_circular(G, with_labels=True, node_color=node_colors, node_size=3000, cmap=plt.cm.rainbow)
#    plt.figure()
    plt.show()
    print('\n',color_map)
    print('\n',node_colors)
    print('\n\n')
# Plot the lowest-energy sample if it meets the constraints
sample = next(response.samples())
if not csp.check(sample):
    print("Failed to color map")
else:
    plot_map(sample)
    
####################################################################
bqm_ising=bqm.to_ising()
h=bqm_ising[0]
J=bqm_ising[1]

hnp=h
Jnp=J

h,J=dok2mat(h,J,convert2Tensor=True)
norms=h.norm()*J.norm()
J=J/norms
h=h/norms


#num_reads=250
#sweeps=10

#op=MLOptim(H=h,J=J)
def convertCompatible(h,J):
    h=dict(zip(range(len(h)),h))
    J=sparse.dok_matrix(J)
    J=dict(zip(J.keys(),J.values()))
    return h,J

def ising(sigma,h,J):
    h,J=convertCompatible(h,J)
    e=dimod.ising_energy(sigma,h,J)
    return e

def discrete(q):
    q=np.array(q)
    q[q>0]=1 
    q[q<=0]=-1
    q=[int(i) for i in q]
    return list(q)

hNumpy,JNumpy=dok2mat(hnp,Jnp)
print('Processing ML')
e=[]
s=[]

num_reads=500
sweeps=10


for j in range(num_reads):       
#        mlSigma,mlEnergy,H_,J_,allSigma=MLOptim(H=hTensor,J=JTensor,sweeps=sweeps)#,sigma=sigma)
    mlOutput=MLOptim(H=h,J=J,sweeps=sweeps)
    allSigma=mlOutput['allSigma']
    
    print('Running for num_reads : ',j)
    
    
    allMLSigma=[discrete(allSigma[i]) for i in range(len(allSigma))]
    allMLEnergy=[ising(allMLSigma[i],hNumpy,JNumpy) for i in range(len(allSigma))]    

    s.append(allMLSigma)
    e.append(allMLEnergy)
    
e=np.array(e)
minE=np.min(e)
ind = np.unravel_index(np.argmin(e), e.shape)
sig=s[ind[0]][ind[1]]

sigma=[int((i+1)/2) for i in sig]
sk=list(sample.keys())

print('Printing for MLOptim')
sd=dict(zip(sk,sigma))
plt.figure()
plot_map(sd)


k=np.where(e==minE)

if len(k[0])>1:
    print('\n{} other solution found as well...\n\n'.format(len(k[0])-1)) 
    
print(ind,minE)

plt.figure();plt.boxplot(e); plt.grid()
###########################################

#sigma=op['allSigma']
#l=[]
#for i in sigma:
#    s=np.array(i)
#    th=0
#    s[s>th]=1
#    s[s<th]=0
#    
#    s=[int(i) for i in s]
#    l.append(s)
#    
#sk=list(sample.keys())
#print('Printing for MLOptim')
#for s in l:
#    sd=dict(zip(sk,s))
#    
#    plt.figure()
#    plot_map(sd)

##########################################
# SAMPLER NEAL
#
#sampler=neal.SimulatedAnnealingSampler()
#
#response = sampler.sample_ising(hnp, Jnp)
#response.change_vartype('BINARY')
#sample = next(response.samples())
#plt.figure()
#if not csp.check(sample):
#    print("Failed to color map")
#else:
#    plot_map(sample)