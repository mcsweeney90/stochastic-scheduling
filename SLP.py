#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic longest path.
"""

import networkx as nx
import numpy as np
from scipy.stats import norm
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
    def __init__(self, mu=0.0, var=0.0): 
        self.mu = mu
        self.var = var
        self.sd = sqrt(var)
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


class SDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is an NetworkX digraph with RV nodes and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
    
    def monte_carlo(self, samples, dist="NORMAL", return_paths=False):
        """
        Monte Carlo method to estimate the distribution of the longest path. 
        TODO: return paths.
        """   
        
        E = []
        for _ in range(samples):
            L = {}
            for t in self.top_sort:
                w = self.graph.nodes[t]['weight'].realize(dist=dist) 
                st = 0.0
                parents = list(self.graph.predecessors(t))
                for p in parents:
                    pst = L[p]
                    try:
                        pst += self.graph[p][t]['weight'].realize(dist=dist)
                    except AttributeError: # Disjunctive edge.
                        pass
                    st = max(st, pst)
                L[t] = st + w
            E.append(L[self.top_sort[-1]])  # Assumes single exit task.
        return E
    
    def np_mc(self, samples, dist="NORMAL"):
        """
        Numpy version of MC.
        TODO: - Memory limit assumes 16GB RAM, check Matt's machine.
        TODO: no check if positive!
        """
                
        x = self.size * samples
        mem_limit = 1800000000
        if x < mem_limit:        
            L = {}
            for t in self.top_sort:
                m, s = self.graph.nodes[t]['weight'].mu, self.graph.nodes[t]['weight'].sd
                if dist in ["N", "NORMAL", "normal"]:  
                    w = np.random.normal(m, s, samples)
                elif dist in ["G", "GAMMA", "gamma"]:
                    v = self.graph.nodes[t]['weight'].var
                    sh, sc = (m * m)/v, v/m
                    w = np.random.gamma(sh, sc, samples)
                elif dist in ["U", "UNIFORM", "uniform"]:
                    u = sqrt(3) * s
                    w = np.random.uniform(-u + m, u + m, samples) 
                parents = list(self.graph.predecessors(t))
                if not parents:
                    L[t] = w 
                    continue
                pmatrix = []
                for p in self.graph.predecessors(t):
                    try:
                        m, s = self.graph[p][t]['weight'].mu, self.graph[p][t]['weight'].sd
                        if dist in ["N", "NORMAL", "normal"]: 
                            e = np.random.normal(m, s, samples)
                        elif dist in ["G", "GAMMA", "gamma"]:
                            v = self.graph[p][t]['weight'].var
                            sh, sc = (m * m)/v, v/m
                            e = np.random.gamma(sh, sc, samples)
                        elif dist in ["U", "UNIFORM", "uniform"]:
                            u = sqrt(3) * s
                            e = np.random.uniform(-u + m, u + m, samples)  
                        pmatrix.append(np.add(L[p], e))
                    except AttributeError:
                        pmatrix.append(L[p])
                st = np.amax(pmatrix, axis=0)
                L[t] = np.add(w, st)
            return L[self.top_sort[-1]] 
        else:
            E = []
            mx_samples = mem_limit//self.size
            runs = samples//mx_samples
            # print(mx_samples, runs)
            extra = samples % mx_samples
            for _ in range(runs):
                E += list(self.np_mc(samples=mx_samples, dist=dist))
            E += list(self.np_mc(samples=extra, dist=dist))
            return E 
    
    def CPM(self, variance=False):
        """
        Returns the classic PERT-CPM bound on the expected value of the longest path.
        If variance == True, also returns the variance of this path to use as a rough estimate
        of the longest path variance.
        """
        C = {}       
        for t in self.top_sort:
            st = 0.0
            if variance:
                v = 0.0
            for p in self.graph.predecessors(t):
                pst = C[p] if not variance else C[p].mu
                try:
                    pst += self.graph[p][t]['weight'].mu
                except AttributeError: # Disjunctive edge.
                    pass
                st = max(st, pst)  
                if variance and st == pst:
                    v = C[p].var
                    try:
                        v += self.graph[p][t]['weight'].var
                    except AttributeError:
                        pass
            m = self.graph.nodes[t]['weight'].mu
            if not variance:
                C[t] = m + st      
            else:
                var = self.graph.nodes[t]['weight'].var
                C[t] = RV(m + st, var + v)                                    
        return C    
    
    def kamburowski(self):
        """
        Returns:
            - lm, lower bounds on the mean. Dict in the form {task ID : m_underline}.
            - um, upper bounds on the mean. Dict in the form {task ID : m_overline}.
            - ls, lower bounds on the variance. Dict in the form {task ID : s_underline}.
            - us, upper bounds on the variance. Dict in the form {task ID : s_overline}.
        """
        lm, um, ls, us = {},{}, {}, {}
        for t in self.top_sort:
            nw = self.graph.nodes[t]['weight']
            parents = list(self.graph.predecessors(t))
            # Entry task(s).
            if not parents:
                lm[t], um[t] = nw.mu, nw.mu
                ls[t], us[t] = nw.var, nw.var
                continue
            # Lower bound on variance.
            if len(parents) == 1:
                ls[t] = ls[parents[0]] + nw.var
                try:
                    ls[t] += self.graph[parents[0]][t]['weight'].var
                except AttributeError:
                    pass
            else:
                ls[t] = 0.0
            # Upper bound on variance.
            v = 0.0
            for p in parents:
                sv = us[p] + nw.var
                try:
                    sv += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                v = max(v, sv)
            us[t] = v
            # Lower bound on mean.
            Xunder = []
            for p in parents:
                pmu = lm[p] + nw.mu
                pvar = ls[p] + nw.var
                try:
                    pmu += self.graph[p][t]['weight'].mu
                    pvar += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                Xunder.append(RV(pmu, pvar))
            Xunder = list(sorted(Xunder, key=lambda x:x.var))
            lm[t] = funder(Xunder)
            # Upper bound on mean.
            Xover = []
            for p in parents:
                pmu = um[p] + nw.mu
                pvar = us[p] + nw.var
                try:
                    pmu += self.graph[p][t]['weight'].mu
                    pvar += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                Xover.append(RV(pmu, pvar))
            Xover = list(sorted(Xover, key=lambda x:x.var))
            um[t] = fover(Xover)
        
        return lm, um, ls, us      

    def sculli(self, direction="downward"):
        """
        Sculli's method for estimating the makespan of a fixed-cost stochastic DAG.
        'The completion time of PERT networks,'
        Sculli (1983).    
        """
        
        L = {}
        if direction == "downward":
            for t in self.top_sort:
                parents = list(self.graph.predecessors(t))
                try:
                    p = parents[0]
                    m = self.graph[p][t]['weight'] + L[p] 
                    for p in parents[1:]:
                        m1 = self.graph[p][t]['weight'] + L[p]
                        m = clark(m, m1, rho=0)
                    L[t] = m + self.graph.nodes[t]['weight']
                except IndexError:  # Entry task.
                    L[t] = self.graph.nodes[t]['weight']  
        elif direction == "upward":
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:
                children = list(self.graph.successors(t))
                try:
                    s = children[0]
                    m = self.graph[t][s]['weight'] + self.graph.nodes[s]['weight'] + L[s] 
                    for s in children[1:]:
                        m1 = self.graph[t][s]['weight'] + self.graph.nodes[s]['weight'] + L[s]
                        m = clark(m, m1, rho=0)
                    L[t] = m  
                except IndexError:  # Entry task.
                    L[t] = 0.0   
        return L
            
    
    def corLCA(self, direction="downward", return_correlation_info=False):
        """
        CorLCA heuristic for estimating the makespan of a fixed-cost stochastic DAG.
        'Correlation-aware heuristics for evaluating the distribution of the longest path length of a DAG with random weights,' 
        Canon and Jeannot (2016).     
        Assumes single entry and exit tasks. 
        This is a fast version that doesn't explicitly construct the correlation tree.
        TODO: make sure upward version works.
        """    
        
        # Dominant ancestors dict used instead of DiGraph for the common ancestor queries. 
        # L represents longest path estimates. V[task ID] = variance of longest path of dominant ancestors (used to estimate rho).
        dominant_ancestors, L, V = {}, {}, {}
        
        if direction == "downward":      
            for t in self.top_sort:     # Traverse the DAG in topological order. 
                nw = self.graph.nodes[t]['weight']
                dom_parent = None 
                for parent in self.graph.predecessors(t):
                    pst = self.graph[parent][t]['weight'] + L[parent]   
                                        
                    # First parent.
                    if dom_parent is None:
                        dom_parent = parent 
                        dom_parent_ancs = set(dominant_ancestors[dom_parent])
                        dom_parent_sd = V[dom_parent]
                        try:
                            dom_parent_sd += self.graph[dom_parent][t]['weight'].var
                        except AttributeError:
                            pass
                        dom_parent_sd = sqrt(dom_parent_sd) 
                        st = pst
                        
                    # At least two parents, so need to use Clark's equations to compute eta.
                    else:                    
                        # Find the lowest common ancestor of the dominant parent and the current parent.
                        for a in reversed(dominant_ancestors[parent]):
                            if a in dom_parent_ancs:
                                lca = a
                                break
                            
                        # Estimate the relevant correlation.
                        parent_sd = V[parent]
                        try:
                            parent_sd += self.graph[parent][t]['weight'].var
                        except AttributeError:
                            pass
                        parent_sd = sqrt(parent_sd) 
                        r = V[lca] / (dom_parent_sd * parent_sd)
                            
                        # Find dominant parent for the maximization.
                        if pst.mu > st.mu: 
                            dom_parent = parent
                            dom_parent_ancs = set(dominant_ancestors[parent])
                            dom_parent_sd = parent_sd
                        
                        # Compute eta.
                        st = clark(st, pst, rho=r)  
                
                if dom_parent is None: # Entry task...
                    L[t] = nw  
                    V[t] = nw.var
                    dominant_ancestors[t] = [t]
                else:
                    L[t] = nw + st 
                    V[t] = dom_parent_sd**2 + nw.var
                    dominant_ancestors[t] = dominant_ancestors[dom_parent] + [t] 
                    
        elif direction == "upward":
            backward_traversal = list(reversed(self.top_sort)) 
            for t in backward_traversal:    
                dom_child = None 
                for child in self.graph.successors(t):
                    cw = self.graph.nodes[child]['weight']
                    cst = self.graph[t][child]['weight'] + cw + L[child]  
                    if dom_child is None:
                        dom_child = child 
                        dom_child_descs = set(dominant_ancestors[dom_child])
                        dom_child_sd = V[dom_child] + self.graph.nodes[dom_child]['weight'].var
                        try:
                            dom_child_sd += self.graph[t][dom_child]['weight'].var
                        except AttributeError:
                            pass
                        dom_child_sd = sqrt(dom_child_sd) 
                        st = cst
                    else: 
                        for a in reversed(dominant_ancestors[child]):
                            if a in dom_child_descs:
                                lca = a
                                break
                        child_sd = V[child] + self.graph.nodes[child]['weight'].var 
                        try:
                            child_sd += self.graph[t][child]['weight'].var
                        except AttributeError:
                            pass
                        child_sd = sqrt(child_sd) 
                        # Find LCA task. TODO: don't like this, rewrite so not necessary.
                        for s in self.top_sort:
                            if s == lca:
                                lca_var = self.graph.nodes[s]['weight'].var  
                                break
                        r = (V[lca] + lca_var) / (dom_child_sd * child_sd) 
                        if cst.mu > st.mu: 
                            dom_child = child
                            dom_child_descs = set(dominant_ancestors[child])
                            dom_child_sd = child_sd
                        st = clark(st, cst, rho=r)  
                if dom_child is None: # Entry task...
                    L[t], V[t] = 0.0, 0.0
                    dominant_ancestors[t] = [t]
                else:
                    L[t] = st 
                    V[t] = dom_child_sd**2 
                    dominant_ancestors[t] = dominant_ancestors[dom_child] + [t]            
        
        if return_correlation_info:
            return L, dominant_ancestors, V
        return L     
    
    def number_of_paths(self):
        """
        Count the number of paths through DAG.
        (Typically only used to show how enormous and impractical it is.)
        """        
        paths = {}
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            if not parents:
                paths[t] = 1
            else:
                paths[t] = sum(paths[p] for p in parents)                
        return paths       
    
def clark(r1, r2, rho=0, minimization=False):
    """
    Returns a new RV representing the maximization of self and other whose mean and variance
    are computed using Clark's equations for the first two moments of the maximization of two normal RVs.
    
    See:
    'The greatest of a finite set of random variables,'
    Charles E. Clark (1983).
    """
    a = np.sqrt(r1.var + r2.var - 2 * r1.sd * r2.sd * rho)     
    b = (r1.mu - r2.mu) / a            
    cdf = norm.cdf(b)
    mcdf = norm.cdf(-b)
    pdf = norm.pdf(b)   
    if minimization:
        mu = r1.mu * mcdf + r2.mu * cdf - a * pdf 
        var = (r1.mu**2 + r1.var) * mcdf
        var += (r2.mu**2 + r2.var) * cdf
        var -= (r1.mu + r2.mu) * a * pdf
        var -= mu**2 
    else:
        mu = r1.mu * cdf + r2.mu * mcdf + a * pdf      
        var = (r1.mu**2 + r1.var) * cdf
        var += (r2.mu**2 + r2.var) * mcdf
        var += (r1.mu + r2.mu) * a * pdf
        var -= mu**2         
    return RV(mu, var) 