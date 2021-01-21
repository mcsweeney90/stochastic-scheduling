#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script.
"""

import os, dill
from timeit import default_timer as timer
from src import RV

size = 100
stg_dag_path = 'graphs/STG/{}'.format(size)

# start = timer()
# for dname in os.listdir(stg_dag_path): 
#     print("\nDAG: {}".format(dname[:-5]))
#     with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
#         T = dill.load(file)
#     T.set_weights(n_processors=8, cov=0.2)    
#     heft_schedule = T.get_averaged_schedule(heuristic="HEFT")
#     heft = heft_schedule.longest_path(method="S")
#     print("HEFT length: {}".format(heft))
#     sheft_schedule = T.get_averaged_schedule(heuristic="HEFT", avg_type="SHEFT")
#     sheft = sheft_schedule.longest_path(method="S")
#     print("SHEFT length: {}".format(sheft))
#     sdls_schedule = T.SDLS()
#     sdls = sdls_schedule.longest_path(method="C")
#     print("SDLS length: {}".format(sdls))
# elapsed = timer() - start
# print("Time taken: {}".format(elapsed))
    
with open('{}/147.dill'.format(stg_dag_path), 'rb') as file:
    T = dill.load(file)

T.set_weights(n_processors=4, cov=0.1)
mean = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.mu
exp_comm, exp_comp = 0.0, 0.0
for t in T.top_sort:
    node_weights = list(v.mu for v in T.graph.nodes[t]['weight'].values())
    exp_comp += sum(node_weights)/len(node_weights)
    for s in T.graph.successors(t):
        edge_weights = list(mean(v) for v in T.graph[t][s]['weight'].values())
        edge_weights += edge_weights
        edge_weights += [0.0]*4
        exp_comm += sum(edge_weights)/len(edge_weights)
    
print(exp_comp, exp_comm)

