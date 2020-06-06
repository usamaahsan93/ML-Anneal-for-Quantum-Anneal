#CODE FROM https://docs.ocean.dwavesys.com/en/latest/examples/map_coloring.html
import dwavebinarycsp
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import matplotlib.pyplot as plt

import torch
from SAVsMLConvergence import MLOptim
from factorFunctions import dok2mat
import neal
import numpy as np
plt.close('all')
# Represent the map as the nodes and edges of a graph
provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE',
             'QC', 'SK', 'YT']
neighbors = [('AB', 'BC'), ('AB', 'NT'), ('AB', 'SK'), ('BC', 'NT'), ('BC', 'YT'),
             ('MB', 'NU'), ('MB', 'ON'), ('MB', 'SK'), ('NB', 'NS'), ('NB', 'QC'),
             ('NL', 'QC'), ('NT', 'NU'), ('NT', 'SK'), ('NT', 'YT'), ('ON', 'QC')]

# Function for the constraint that two nodes with a shared edge not both select
# one color
def not_both_1(v, u):
    return not (v and u)

# Valid configurations for the constraint that each node select a single color
one_color_configurations = {(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)}
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

#sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi', token='Enter Token Here', solver='DW_2000Q_2_1'))
sampler=neal.SimulatedAnnealingSampler()
response = sampler.sample(bqm, num_reads=50)

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
op=MLOptim(H=h,J=J)

sigma=op['allSigma']
l=[]
for i in sigma:
    s=np.array(i)
    th=0
    s[s>th]=1
    s[s<th]=0
    
    s=[int(i) for i in s]
    l.append(s)
    
sk=list(sample.keys())
print('Printing for MLOptim')
for s in l:
    sd=dict(zip(sk,s))
    
    plt.figure()
    plot_map(sd)


sampler=neal.SimulatedAnnealingSampler()

response = sampler.sample_ising(hnp, Jnp)
response.change_vartype('BINARY')
sample = next(response.samples())
plt.figure()
if not csp.check(sample):
    print("Failed to color map")
else:
    plot_map(sample)
