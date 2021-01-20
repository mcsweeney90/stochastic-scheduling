#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script.
"""

import os, dill
from timeit import default_timer as timer
# from src import TDAG

size = 100
stg_dag_path = 'graphs/STG/{}'.format(size)

start = timer()
for dname in os.listdir(stg_dag_path): 
    print("\nDAG: {}".format(dname[:-5]))
    with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
        T = dill.load(file)
    T.set_weights(n_processors=8, cov=0.5)    
    heft_schedule = T.get_averaged_schedule(heuristic="HEFT")
    heft = heft_schedule.longest_path(method="S")
    print("HEFT length: {}".format(heft))
    sheft_schedule = T.get_averaged_schedule(heuristic="HEFT", avg_type="SHEFT")
    sheft = sheft_schedule.longest_path(method="S")
    print("SHEFT length: {}".format(sheft))
    sdls_schedule = T.SDLS()
    sdls = sdls_schedule.longest_path(method="S")
    print("SDLS length: {}".format(sdls))
elapsed = timer() - start
print("Time taken: {}".format(elapsed))
    
# with open('{}/147.dill'.format(stg_dag_path), 'rb') as file:
#     T = dill.load(file)
# start = timer()
# T.set_weights(n_processors=4, cov=0.1)
# A = T.get_averaged_graph(avg_type="MEAN")
# U = A.get_upward_ranks(weighted=True)
# print(U)
# # for t in A.top_sort:
# #     print(t, A.graph.nodes[t]['weight'])
    
# elapsed = timer() - start
# print("Time taken: {}".format(elapsed))