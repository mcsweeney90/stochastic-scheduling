#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic scheduling simulator.
"""

import random
import networkx as nx
from math import sqrt
from Average import ADAG
from SLP import RV, SDAG
               
class TDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is a NetworkX digraph with {Processor ID : RV} node and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def set_weights(self, n_processors, ccr, cov):
        """
        Used for setting weights for DAGs from the STG.
        """
        for t in self.top_sort:
            self.graph.nodes[t]['weight'] = {}
            # Set possible node weights...
            for p in self.graph.predecessors(t):
                self.graph[p][t]['weight'] = {}
                # Set possible edge weights...  
                
    def get_average_graph(self, avg_type="MEAN"):
        A = self.graph.__class__()
        A.add_nodes_from(self.graph)
        A.add_edges_from(self.graph.edges)
        for t in self.top_sort:
            A.nodes[t]['weight'] = {k : v.average(avg_type=avg_type) for k, v in self.graph.nodes[t]['weight'].items()}
            for s in self.graph.successors(t):
                A[t][s]['weight'] = {}
                average = lambda r :0.0 if (type(r) == float or type(r) == int) else r.average(avg_type=avg_type)
                A[t][s]['weight'] = {k : average(v) for k, v in self.graph[t][s]['weight'].items()}
                # for k, v in self.graph[t][s]['weight'].items():
                #     A[t][s]['weight'][k] = 0.0 if type(v) == float else v.average(avg_type=avg_type) 
        return ADAG(A)
    
    def schedule_to_graph(self, schedule, where):
        S = self.graph.__class__()
        S.add_nodes_from(self.graph)
        S.add_edges_from(self.graph.edges)
        # Set the weights.
        for t in self.top_sort:
            w = where[t]
            S.nodes[t]['weight'] = self.graph.nodes[t]['weight'][w] 
            for s in self.graph.successors(t):
                w1 = where[s]
                S[t][s]['weight'] = self.graph[t][s]['weight'][(w, w1)] # TODO: make sure zero if they're the same.
            # Add disjunctive edge if necessary.
            idx = list(r[0] for r in schedule[w]).index(t)
            # idx = schedule[w].index(t) # TODO: loads not in this form
            if idx > 0:
                d = schedule[w][idx - 1][0]
                if not S.has_edge(d, t):
                    S.add_edge(d, t)
                    S[d][t]['weight'] = 0.0
        return SDAG(S) 
    
    def get_average_schedule(self, heuristic="HEFT", avg_type="MEAN"):
        avg_graph = self.get_average_graph(avg_type=avg_type)
        if heuristic == "HEFT": 
            pi, W = avg_graph.HEFT()
        elif heuristic == "HEFT-WM":
            pi, W = avg_graph.HEFT(weighted=True)
        # Construct the schedule graph.
        schedule_graph = self.schedule_to_graph(schedule=pi, where=W)
        return schedule_graph
    
    # def HEFT(self, weighted=False):
    #     # Get the average graph.
    #     avg_graph = self.get_average_graph(avg_type="MEAN")
    #     # Apply HEFT and get the schedule. 
    #     pi, W = avg_graph.HEFT(weighted=weighted)
    #     # Construct the schedule graph.
    #     schedule_graph = self.schedule_to_graph(schedule=pi, where=W)
    #     return schedule_graph
    
         
        

        
                
