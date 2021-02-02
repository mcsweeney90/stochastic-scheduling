#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script.
"""

import os, dill
import numpy as np
from timeit import default_timer as timer
from functools import partial
from src import RV, SSTAR, SDLS, RobHEFT, HEFT, MCS, PEFT

size = 100
stg_dag_path = 'graphs/STG/{}'.format(size)

# start = timer()
# slr = {"HEFT": [], "PEFT" : []}
# for dname in os.listdir(stg_dag_path): 
#     # print("\nDAG: {}".format(dname[:-5]))
#     with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
#         T = dill.load(file)
#     T.set_weights(n_processors=4, cov=0.3) 
    
#     A = T.get_scalar_graph(kind="A", avg_type="MEAN")
    
#     lb = A.makespan_lower_bound()
    
#     pi, _ = HEFT(A)
#     slr["HEFT"].append(max(L[-1][2] for L in pi.values())/lb)
#     # print(max(L[-1][2] for L in pi.values()))
    
#     pi1, _ = PEFT(A)
#     slr["PEFT"].append(max(L[-1][2] for L in pi1.values())/lb)
#     # print(max(L[-1][2] for L in pi1.values()))
    
# elapsed = timer() - start
# print("Time taken: {}".format(elapsed))
    
with open('{}/147.dill'.format(stg_dag_path), 'rb') as file:
    T = dill.load(file)
T.set_weights(n_processors=4, cov=0.5) 

# mst = T.minimal_serial_time(mc_samples=100000)
# print("MST: RV(mu = {}, var = {})".format(sum(mst)/len(mst), np.var(mst)))

heft_schedule = SSTAR(T, det_heuristic=HEFT)
heft = heft_schedule.longest_path(method="S")
print("HEFT length: {}".format(heft))

# heft_schedule = SSTAR1(T, det_heuristic=HEFT)
# heft = heft_schedule.longest_path(method="S")
# print("HEFT length: {}".format(heft))

# HEFT_WM = partial(HEFT, weighted=True)
# wheft_schedule = SSTAR(T, det_heuristic=HEFT_WM)
# wheft = wheft_schedule.longest_path(method="S")
# print("WHEFT length: {}".format(wheft))

# sheft_schedule = SSTAR(T, det_heuristic=HEFT, avg_type="SHEFT")
# sheft = sheft_schedule.longest_path(method="S")
# print("SHEFT length: {}".format(sheft))

# s_func = lambda r : r.mu + r.sd if r.mu > r.sd else r.mu + (r.sd/r.mu)
# sheft_schedule = SSTAR1(T, det_heuristic=HEFT, scal_func=s_func)
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

# start = timer()
# rob_schedule = RobHEFT(T, alpha=90)
# elapsed = timer() - start

# rob = rob_schedule.longest_path(method="S")
# print("RobHEFT length: {}".format(rob))
# print("Time taken: {}".format(elapsed))

mcs_schedule = MCS(T)
mcs = mcs_schedule.longest_path(method="S")
print("MCS length: {}".format(mcs))



# L = []
# for _ in range(10):
#     # Get the standard HEFT schedule.
#     avg_graph = T.get_scalar_graph(kind="A", avg_type="MEAN")
#     mean_static_schedule, where = HEFT(avg_graph) 
#     omega_mean = T.schedule_to_graph(schedule=mean_static_schedule, where_scheduled=where)
#     if all(omega_mean.graph.edges != s.graph.edges for s in L):
#         L.append(omega_mean)
# print(len(L))