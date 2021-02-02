#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upward ranking-like priorities.

Two basic approaches here: 1. Scalarize weights, then compute upward rank.
                           2. Compute stochastic upward ranks, then scalarize.
"""

import os, dill, sys
import numpy as np
from timeit import default_timer as timer
sys.path.append('../') 
from src import HEFT

def mean_only(est_makespans):
    return min(est_makespans, key=lambda m : m.mu)

def UCB(est_makespans, c=1):
    return min(est_makespans, key=lambda m : m.mu + c * m.sd)

# Scalarization functions.
mean = lambda r : r.mu
sheft = lambda r : r.mu + r.sd if r.mu > r.sd else r.mu + (r.sd/r.mu)
ucb1 = lambda r : r.mu + r.sd
ucb2 = lambda r : r.mu + 2*r.sd
ucb3 = lambda r : r.mu + 3*r.sd
pr = lambda r: r.mu*r.sd  
div = lambda r: r.mu/r.sd # TODO: would it not make more sense to prioritize small values? 
scal_functions = [mean, sheft, ucb1, ucb2, ucb3, pr, div]


size = 100
stg_dag_path = '../graphs/STG/{}'.format(size)

with open('{}/147.dill'.format(stg_dag_path), 'rb') as file:
    T = dill.load(file)
T.set_weights(n_processors=4, cov=0.3) 

A = T.get_scalar_graph()

for i, scal_func in enumerate(scal_functions):
    print("\n{}".format(i))
    G = T.get_scalar_graph(scal_func)
    # Non-weighted.
    prios = G.get_upward_ranks(weighted=False)
    P, where = A.list_scheduling(priorities=prios, policy="EFT")
    S = T.schedule_to_graph(schedule=P, where_scheduled=where)
    dist = S.longest_path(method="MC", mc_dist="G", mc_samples=1000)
    mu = sum(dist)/len(dist)
    var = np.var(dist)
    print("NW: mu = {}, var = {}".format(mu, var))
    
    
    # Weighted.
    wprios = G.get_upward_ranks(weighted=True)
    P, where = A.list_scheduling(priorities=wprios, policy="EFT")
    S = T.schedule_to_graph(schedule=P, where_scheduled=where)
    dist = S.longest_path(method="MC", mc_dist="G", mc_samples=1000)
    mu = sum(dist)/len(dist)
    var = np.var(dist)
    print("W: mu = {}, var = {}".format(mu, var))
    



