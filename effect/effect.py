#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Illustrate the effect of stochasticity.
TODO: find an intuitive way to emphasize the importance of taking stochasticity into account.
"""

import os, dill, sys
import numpy as np
from timeit import default_timer as timer
sys.path.append('../') 
from src import SSTAR

size = 100
stg_dag_path = '../graphs/STG/{}'.format(size)

with open('{}/147.dill'.format(stg_dag_path), 'rb') as file:
    T = dill.load(file)
T.set_weights(n_processors=4, cov=0.01) # Initial variances are not used.

H = SSTAR(T, weighted=False)
for cov in [0.01, 0.1, 0.5]:
    print("\nCoV = {}".format(cov))
    # Re-write costs of H.
    H.adjust_uncertainty(new_cov=cov)
    # Compute the empirical distribution.
    emp_dist = H.longest_path(method="MC", mc_dist="GAMMA", mc_samples=1000)
    mu = sum(emp_dist)/len(emp_dist)
    var = np.var(emp_dist)
    print("Schedule length: mu = {}, var = {}".format(mu, var))
    mx, mn = max(emp_dist), min(emp_dist)
    print("Max = {}, min = {}".format(mx, mn))




