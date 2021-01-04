#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic scheduling simulator.
"""

import random
import numpy as np
import networkx as nx
from math import sqrt

class RV:
    """
    Random variable class.
    Notes:
        - Defined by only mean and variance so can in theory be from any distribution but some functions
          e.g., addition and multiplication assume (either explicitly or implicitly) RV is Gaussian.
          (Addition/mult only done when RV represents a finish time/longest path estimate so is assumed to be
          at least roughly normal anyway.)
        - ID attribute is often useful (e.g., for I/O).
        - Doesn't check that self.mu and self.var are nonzero for gamma. Obviously shouldn't be but sometimes tempting
          programmatically.
        - Random.random faster than numpy for individual realizations.
    """
    def __init__(self, mu=0.0, var=0.0, ID=None): 
        self.mu = mu
        self.var = var
        self.sd = sqrt(var)
        self.ID = ID
    def __repr__(self):
        return "RV(mu = {}, var = {})".format(self.mu, self.var)
    # Overload addition operator.
    def __add__(self, other): 
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu + other, self.var)
        return RV(self.mu + other.mu, self.var + other.var) 
    __radd__ = __add__ 
    # Overload subtraction operator.
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu - other, self.var)
        return RV(self.mu - other.mu, self.var + other.var)
    __rsub__ = __sub__ 
    # Overload multiplication operator.
    def __mul__(self, c):
        return RV(c * self.mu, c * c * self.var)
    __rmul__ = __mul__ 
    # Overload division operators.
    def __truediv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rtruediv__ = __truediv__ 
    def __floordiv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rfloordiv__ = __floordiv__     
    def reset(self):
        """Set all attributes except ID to their defaults."""
        self.mu, self.var, self.sd = 0.0, 0.0, 0.0    
    def realize(self, dist, static=False):
        if static:
            return self.mu 
        elif dist in ["N", "NORMAL", "normal"]:    
            r = random.gauss(self.mu, self.sd)    # Faster than numpy for individual realizations.         
            return r if r > 0.0 else -r
        elif dist in ["G", "GAMMA", "gamma"]:
            return random.gammavariate(alpha=(self.mu**2 / self.var), beta=self.var/self.mu)      
        elif dist in ["U", "UNIFORM", "uniform"]:
            u = sqrt(3) * self.sd
            r = self.mu + random.uniform(-u, u)                
            return r if r > 0.0 else -r 
    def average(self, avg_type="MEAN"):
        if avg_type == "MEAN":
            return self.mu
        elif avg_type == "UCB":
            return self.mu + self.sd
        elif avg_type == "SHEFT": # TODO: check.
            return self.mu + self.sd if self.mu > self.sd else self.mu + (self.sd/self.mu)
        
class TDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is a NetworkX digraph with {Processor ID : RV} node and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def set_weights(self, n_processors, ccr):
        """
        Used for setting weights for DAGs from the STG.
        """
        for t in self.top_sort:
            self.graph.nodes[t]['weight'] = {}
            # Set possible node weights...
            for p in self.graph.predecessors(t):
                self.graph[p][t]['weight'] = {}
                # Set possible edge weights...
                
    def upward_ranks(self, heuristic):
        """TODO. Want HEFT, HEFT-WM and SHEFT at least."""
        if heuristic == "HEFT":
            ranks = {}
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:
                ranks[t] = np.mean(v.mu for v in self.graph.nodes[t]['weight'].values())
                try:
                    ranks[t] += max(np.mean(self.graph[t][s]['weight'].values()) + ranks[s] for s in self.graph.successors(t))
                except ValueError:
                    pass   
            return ranks
        
    def simulate(self, priority_list, expected=True):
        EFT, where_scheduled = {}, {}
        workers = self.graph.nodes[self.top_sort[0]]['weight'].keys()
        loads = {w:[] for w in workers}
        for t in priority_list:
            worker_finish_times = {}
            for w in workers:
                task_cost = self.graph.nodes[t]['weight'][w].mu
                drt = max(EFT[p] + self.graph[p][t]['weight'][(where_scheduled[p], w)] for p in self.graph.predecessors(t)) # What if entry?
                # Find time worker can actually execute the task.
                worker_finish_times[w] = earliest_finish_time(loads[w], drt, task_cost)
            min_worker = min(workers, key=lambda w:worker_finish_times[w][0])
            where_scheduled[t] = min_worker            
            ft, idx = worker_finish_times[min_worker]
            EFT[t] = ft
            st = ft - self.graph.nodes[t]['weight'][min_worker].mu        
            # Add to load.           
            if not loads[min_worker] or idx < 0:             
                loads[w].append((t, st, ft))  
            else: 
                loads[w].insert(idx, (t, st, ft)) 
        return loads, where_scheduled          
    
    def get_schedule(self, heuristic):
        # Compute ranks.
        ranks = self.upward_ranks(heuristic)
        # Get priority list.
        priority_list = list(sorted(ranks, key=ranks.get, reverse=True))   
        # Simulate scheduling to estimate processor loads. 
        loads, where_scheduled = self.simulate(priority_list)
        S = nx.DiGraph()
        # Construct schedule graph...
        return S    
        
def earliest_finish_time(load, drt, task_cost):
    # Check if it can be scheduled before any other task in the load.
    prev_finish_time = 0.0
    for i, t in enumerate(load):
        if t[1] < drt:
            prev_finish_time = t[2]
            continue
        poss_finish_time = max(prev_finish_time, drt) + task_cost
        if poss_finish_time <= t[1]:
            return (poss_finish_time, i) 
        prev_finish_time = t[2]    
    # No valid gap found.
    return (task_cost + max(load[-1][2], drt), -1)          # TODO: what if load empty?
        
                
            
    
    
        
# class Task:
#     """
#     Represent tasks.
#     """         
#     def __init__(self, task_type=None):
#         """
#         Create Task object.       
#         """  
        
#         self.type = task_type 
#         self.ID = None    
#         self.entry = False  
#         self.exit = False   
        
#         self.comp_costs = {} 
#         self.comm_costs = {}     
        
#         self.FT = None  
#         self.scheduled = False  
#         self.where_scheduled = None              
    
#     def reset(self):
#         """Resets some attributes to defaults so execution of the task can be simulated again."""
#         self.FT = None   
#         self.scheduled = False
#         self.where_scheduled = None      
        
#     def average_cost(self, avg_type="HEFT"):
#         """
#         Compute the "average" computation time of the Task. 
#         Usually used for setting priorities in HEFT and similar heuristics.
#         """
#         if avg_type == "HEFT":
#             return np.mean(v.mu for v in self.comp_costs.values())
#         elif avg_type == "HEFT-WM" or avg_type == "WM":
#             s = sum(1/v.mu for v in self.comp_costs.values())
#             return len(self.comp_costs) / s 
#         elif avg_type == "SHEFT":
#             return np.mean(v.average(avg_type="SHEFT") for v in self.comp_costs.values())
#         raise ValueError('Unrecognized avg_type specified for average_cost.') 
        
#     def average_comm_cost(self, child, avg_type="HEFT"):
#         """
#         Compute an "average" communication cost between Task and one of its children. 
#         """         
                
#         if avg_type == "HEFT":
#             return np.mean(v.mu for v in self.comm_costs[child.ID].values())      
#         elif avg_type == "HEFT-WM" or avg_type == "WM":
#             s1 = sum(1/v.mu for v in self.comp_costs.values())
#             s2 = sum(1/v.mu for v in child.comp_costs.values())
#             cbar = 0.0
#             for k, v in self.comm_costs[child.ID].items():
#                 t_w = self.comp_costs[k[0]].mu
#                 c_w = child.comp_costs[k[1]].mu                
#                 cbar += v.mu/(t_w * c_w) 
#             cbar /= (s1 * s2)
#             return cbar       
#         elif avg_type == "SHEFT": 
#             return np.mean(v.average(avg_type="SHEFT") for v in self.comm_costs[child.ID].values())            
#         raise ValueError('Unrecognized avg_type specified for average_comm_cost.')  
        
