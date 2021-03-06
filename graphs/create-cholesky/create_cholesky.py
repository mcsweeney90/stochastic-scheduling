#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create stochastic schedule DAGs for Cholesky.
"""

import dill, pathlib, sys, os
import networkx as nx
import numpy as np
import itertools as it
from timeit import default_timer as timer
sys.path.append('../../') 
from src import TDAG, RV

# # Source for topologies.
# topologies = 'cholesky-topologies'
# # Destination for saved files.
# dest = '../cholesky'
# pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
# # Load the timings data.
# with open('skylakeV100_timings.dill', 'rb') as file:
#     timings = dill.load(file)
    
# # Compute means and variances.
# weights = {}
# for nb in [128, 1024]:
#     weights[nb] = {}
#     for adt in ["N", "P"]:
#         weights[nb][adt] = {}
#         for task_type in ["G", "P", "S", "T"]:
#             weights[nb][adt][task_type] = {} 
#             for d in ["C", "G", "CC", "CG", "GC", "GG"]: 
#                 mu = sum(timings[nb][adt][task_type][d])/len(timings[nb][adt][task_type][d])
#                 var = np.var(timings[nb][adt][task_type][d])
#                 weights[nb][adt][task_type][d] = RV(mu, var)

# for name in range(5, 11, 5):   
#     print("\n{}".format(name))
#     with open('{}/{}.dill'.format(topologies, name), 'rb') as file:
#         G = dill.load(file)
#     for nc, ng in [(7, 1), (28, 4)]:
#         print(nc, ng)
#         full_dest = "{}/single".format(dest) if nc == 7 else "{}/multiple".format(dest)
#         pathlib.Path(full_dest).mkdir(parents=True, exist_ok=True)
#         for nb in [128, 1024]:
#             print(nb)
#             for adt in ["N", "P"]:   
#                 print(adt)
#                 C = nx.DiGraph() # Can't just copy since need to relabel...
#                 ids = {}
#                 for i, t in enumerate(nx.topological_sort(G)):
#                     C.add_node(i + 1)
#                     # Get relevant weights...
#                     task_type = t[0]
#                     C.nodes[i + 1]['weight'] = {w : weights[nb][adt][task_type]["C"] for w in range(nc)} 
#                     for w in range(nc, nc + ng):
#                         C.nodes[i + 1]['weight'][w] = weights[nb][adt][task_type]["G"] 
#                     ids[t] = i + 1
#                     for p in G.predecessors(t):
#                         C.add_edge(ids[p], ids[t])
#                         parent_type = p[0]
#                         C[ids[p]][ids[t]]['weight'] = {}
#                         # Seems to work okay... 
#                         for s in range(nc + ng):
#                             for d in range(s + 1, nc + ng):
#                                 if s < nc and d < nc:
#                                     C[ids[p]][ids[t]]['weight'][(s, d)] = 0.0 
#                                 elif s < nc and d >= nc:
#                                     C[ids[p]][ids[t]]['weight'][(s, d)] = weights[nb][adt][task_type]["CG"]
#                                 elif s >= nc:
#                                     C[ids[p]][ids[t]]['weight'][(s, d)] = weights[nb][adt][task_type]["GG"]
                        
#                         # for s, d in it.product(range(nc), range(nc)):
#                         #     C[ids[p]][ids[t]]['weight'][(s, d)] = 0.0  
#                         # for s, d in it.product(range(nc), range(nc, nc + ng)):
#                         #     C[ids[p]][ids[t]]['weight'][(s, d)] = weights[nb][adt][task_type]["CG"] 
#                         # for s, d in it.product(range(nc, nc + ng), range(nc)):
#                         #     C[ids[p]][ids[t]]['weight'][(s, d)] = weights[nb][adt][task_type]["GC"]
#                         # for s, d in it.product(range(nc, nc + ng), range(nc, nc + ng)):
#                         #     C[ids[p]][ids[t]]['weight'][(s, d)] = 0.0 if s == d else weights[nb][adt][task_type]["GG"]
#                 S = TDAG(C)
#                 with open('{}/{}{}{}.dill'.format(full_dest, name, adt, nb), 'wb') as handle: # name[:-5]
#                     dill.dump(S, handle)

with open('../cholesky/single/10N128.dill', 'rb') as file:
    G = dill.load(file) 
start = timer()
# heft = G.get_averaged_schedule(heuristic="HEFT-WM")
sdls = G.SDLS()
# robheft = G.RobHEFT()
elapsed = timer() - start
length = sdls.longest_path(method="C")
print("This took {} seconds".format(elapsed))
print(length)

            

