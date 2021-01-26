#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script.
"""

import os, dill
import numpy as np
from timeit import default_timer as timer
from src import RV, SSTAR, SDLS, RobHEFT

size = 100
stg_dag_path = 'graphs/STG/{}'.format(size)

# start = timer()
# for dname in os.listdir(stg_dag_path): 
#     print("\nDAG: {}".format(dname[:-5]))
#     with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
#         T = dill.load(file)
#     T.set_weights(n_processors=8, cov=0.3)    
#     heft_schedule = T.get_averaged_schedule(heuristic="HEFT")
#     heft = heft_schedule.longest_path(method="S")
#     print("HEFT length: {}".format(heft))
#     sheft_schedule = T.get_averaged_schedule(heuristic="HEFT", avg_type="SHEFT")
#     sheft = sheft_schedule.longest_path(method="S")
#     print("SHEFT length: {}".format(sheft))
#     sdls_schedule = T.SDLS(insertion=None)
#     sdls = sdls_schedule.longest_path(method="C")
#     print("SDLS length: {}".format(sdls))
# elapsed = timer() - start
# print("Time taken: {}".format(elapsed))
    
with open('{}/147.dill'.format(stg_dag_path), 'rb') as file:
    T = dill.load(file)
T.set_weights(n_processors=4, cov=0.1) 

mst = T.minimal_serial_time(mc_samples=100000)
print("MST: RV(mu = {}, var = {})".format(sum(mst)/len(mst), np.var(mst)))

# heft_schedule = SSTAR(T, weighted=False)
# heft = heft_schedule.longest_path(method="S")
# print("HEFT length: {}".format(heft))

# wheft_schedule = SSTAR(T, weighted=True)
# wheft = wheft_schedule.longest_path(method="S")
# print("WHEFT length: {}".format(wheft))

# sheft_schedule = SSTAR(T, avg_type="SHEFT", weighted=False)
# sheft = sheft_schedule.longest_path(method="S")
# print("SHEFT length: {}".format(sheft))

# wsheft_schedule = SSTAR(T, avg_type="SHEFT", weighted=True)
# wsheft = wsheft_schedule.longest_path(method="S")
# print("WSHEFT length: {}".format(wsheft))

# sdls_schedule = SDLS(T, insertion=None)
# sdls = sdls_schedule.longest_path(method="S")
# print("SDLS length: {}".format(sdls))

# isdls_schedule = SDLS(T, insertion="M")
# isdls = isdls_schedule.longest_path(method="S")
# print("ISDLS length: {}".format(isdls))

# rob_schedule = RobHEFT(T, alpha=45)
# rob = rob_schedule.longest_path(method="S")
# print("RobHEFT length: {}".format(rob))
