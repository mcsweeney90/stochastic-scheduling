#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic scheduling simulator.
"""

import random
import networkx as nx
import numpy as np
from scipy.stats import norm
from math import sqrt
from psutil import virtual_memory

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
        """Set all attributes to their defaults."""
        self.mu, self.var, self.sd = 0.0, 0.0, 0.0    
    def realize(self, dist):
        if dist in ["N", "NORMAL", "normal"]:    
            r = random.gauss(self.mu, self.sd)    # Faster than numpy for individual realizations.         
            return r if r > 0.0 else -r
        elif dist in ["G", "GAMMA", "gamma"]:
            return random.gammavariate(alpha=(self.mu**2 / self.var), beta=self.var/self.mu)      
        elif dist in ["U", "UNIFORM", "uniform"]:
            u = sqrt(3) * self.sd
            r = self.mu + random.uniform(-u, u)                
            return r if r > 0.0 else -r 
    def average(self, avg_type="MEAN"):
        if avg_type in ["M", "MEAN"]:
            return self.mu
        elif avg_type in ["U", "UCB"]:
            return self.mu + self.sd
        elif avg_type in ["S", "SHEFT"]: 
            return self.mu + self.sd if self.mu > self.sd else self.mu + (self.sd/self.mu)

class ADAG:
    """Represents a graph with average node and edge weights."""
    def __init__(self, graph):
        """Graph is a NetworkX digraph with {Processor ID : float} node and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def node_mean(self, task, weighted=False):
        if not weighted:
            return sum(self.graph.nodes[task]['weight'].values())/len(self.graph.nodes[task]['weight'])
        s = sum(1/v for v in self.graph.nodes[task]['weight'].values())
        return len(self.graph.nodes[task]['weight']) / s 
    
    def edge_mean(self, parent, child, weighted=False):
        if not weighted:
            return sum(self.graph[parent][child]['weight'].values())/len(self.graph[parent][child]['weight']) 
        s1 = sum(1/v for v in self.graph.nodes[parent]['weight'].values())
        s2 = sum(1/v for v in self.graph.nodes[child]['weight'].values())
        cbar = 0.0
        for k, v in self.graph[parent][child]['weight'].items():
            t_w = self.graph.nodes[parent]['weight'][k[0]]
            c_w = self.graph.nodes[child]['weight'][k[1]]             
            cbar += v/(t_w * c_w) 
        cbar /= (s1 * s2)
        return cbar 
        
    def get_upward_ranks(self, weighted=False):
        ranks = {}
        backward_traversal = list(reversed(self.top_sort))
        for t in backward_traversal:
            ranks[t] = self.node_mean(t, weighted=weighted)
            try:
                ranks[t] += max(self.edge_mean(t, s, weighted=weighted) + ranks[s] for s in self.graph.successors(t))
            except ValueError:
                pass   
        return ranks  
    def get_downward_ranks(self):
        return
    
    def simulate_scheduling(self, priority_list, policy="EFT"):
        """
        Simulates the scheduling of the tasks in priority_list.
        """ 
        # Get list of workers and initialize schedule.
        workers = self.graph.nodes[self.top_sort[0]]['weight'].keys()
        schedule = {w : [] for w in workers}
        # Keep track of finish times and where tasks are scheduled.
        finish_times, where = {}, {}
        # Start the simulation.
        for task in priority_list:
            # print("\n")
            # print(task)
            parents = list(self.graph.predecessors(task))
            # print(list(p for p in parents))
            worker_schedules = {}
            for w in workers:
                task_cost = self.graph.nodes[task]['weight'][w]
                # Find the data-ready time.                    
                drt = 0.0 if not parents else max(finish_times[p] + self.graph[p][task]['weight'][(where[p], w)] for p in parents) 
                # Find time worker can actually execute the task.
                if not schedule[w]:
                    worker_schedules[w] = (drt, drt + task_cost, 0)
                else:
                    found, prev_finish_time = False, 0.0
                    for i, t in enumerate(schedule[w]):
                        if t[1] < drt:
                            prev_finish_time = t[2]
                            continue
                        poss_start_time = max(prev_finish_time, drt) 
                        poss_finish_time = poss_start_time + task_cost
                        if poss_finish_time <= t[1]:
                            found = True
                            worker_schedules[w] = (poss_start_time, poss_finish_time, i)                            
                            break
                        prev_finish_time = t[2]    
                    # No valid gap found.
                    if not found:
                        st = max(schedule[w][-1][2], drt)
                        worker_schedules[w] = (st, st + task_cost, -1)     
            # print(worker_schedules)
            min_worker = min(workers, key=lambda w:worker_schedules[w][1])
            where[task] = min_worker            
            st, ft, idx = worker_schedules[min_worker]
            finish_times[task] = ft   
            # print(finish_times)
            # Add to schedule.           
            if not schedule[min_worker] or idx < 0:             
                schedule[min_worker].append((task, st, ft))  
            else: 
                schedule[min_worker].insert(idx, (task, st, ft)) 
            # print(schedule)
        return schedule, where    
    
    def HEFT(self, weighted=False):
        # Compute upward ranks.
        U = self.get_upward_ranks(weighted=weighted)
        # Sort into priority list.
        priority_list = list(sorted(U, key=U.get, reverse=True))
        # Get schedule.
        return self.simulate_scheduling(priority_list=priority_list)      
    def CPOP(self):
        return
    def PEFT(self):
        return

class SDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is an NetworkX digraph with RV nodes and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
    
    def MC(self, samples, dist="GAMMA"):
        """
        Monte Carlo method to estimate the longest path distribution. 
        Vectorized numpy version which is much faster than the serial approach but definitely a memory hog!
        TODO: Better way to determine memory limit?
        TODO: no check if positive for normal and uniform!
        TODO: modify to determine upward ranks for task prioritization?
        """
        
        # Check if sufficient memory to do entire sampling in single 
        # mem_limit = 1800000000 # TODO: estimate this somehow.  
        mem_limit = virtual_memory().available // 10 # Size of numpy random array ~ 8 * samples
        if self.size*samples < mem_limit:        
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
            extra = samples % mx_samples
            for _ in range(runs):
                E += list(self.MC(samples=mx_samples, dist=dist))
            E += list(self.MC(samples=extra, dist=dist))
            return E       

    def sculli(self, direction="downward"):
        """
        Sculli's method for estimating the makespan of a fixed-cost stochastic DAG.
        'The completion time of PERT networks,'
        Sculli (1983).  
        TODO: upward doesn't include current task cost, which is reverse of convention elsewhere.
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
    
    def corLCA(self, direction="D", return_correlation_info=False):
        """
        CorLCA heuristic for estimating the makespan of a fixed-cost stochastic DAG.
        'Correlation-aware heuristics for evaluating the distribution of the longest path length of a DAG with random weights,' 
        Canon and Jeannot (2016).     
        Assumes single entry and exit tasks. TODO: could be an issue when evaluating partial DAGs during simulated scheduling, keep an eye. 
        This is a fast version that doesn't explicitly construct the correlation tree.
        TODO: make sure upward version works. Add a parameter to control whether or not upward includes current task?
        """    
        
        # Dominant ancestors dict used instead of DiGraph for the common ancestor queries. 
        # L represents longest path estimates. V[task ID] = variance of longest path of dominant ancestors (used to estimate rho).
        dominant_ancestors, L, V = {}, {}, {}
        
        if direction in ["D", "d", "downward", "DOWNWARD"]:      
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
                    
        elif direction in ["U", "u", "UPWARD", "upward"]:
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
                
    def get_averaged_graph(self, stochastic=False, avg_type="MEAN"):
        """Return an equivalent graph with averaged weights."""
        
        # Copy the topology.
        A = self.graph.__class__()
        A.add_nodes_from(self.graph)
        A.add_edges_from(self.graph.edges)
        
        # Scalar averages.
        if not stochastic:
            for t in self.top_sort:
                A.nodes[t]['weight'] = {k : v.average(avg_type=avg_type) for k, v in self.graph.nodes[t]['weight'].items()}
                for s in self.graph.successors(t):
                    A[t][s]['weight'] = {}
                    average = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.average(avg_type=avg_type)
                    A[t][s]['weight'] = {k : average(v) for k, v in self.graph[t][s]['weight'].items()}
            return ADAG(A)
        
        # Stochastic averages.        
        for t in self.top_sort:
            A.nodes[t]['weight'] = stochastic_average(self.graph.nodes[t]['weight'].values(), avg_type=avg_type)
            for s in self.graph.successors(t):
                A[t][s]['weight'] = stochastic_average(self.graph[t][s]['weight'].values(), avg_type=avg_type)
        return SDAG(A)
    
    def get_averaged_schedule(self, heuristic="HEFT", avg_type="MEAN", avg_graph=None):
        """
        Compute a schedule using average costs.
        Scheduled returned in the form of a graph with stochastic costs.
        TODO: assumes sequential/fullahead schedule, how to handle assignment case?
        """
        # Construct the (scalar) "average" graph.
        if avg_graph is None:
            avg_graph = self.get_averaged_graph(avg_type=avg_type)
        
        # Apply specified heuristic. TODO: would ideally like to pass ADAG method as parameter. 
        if heuristic == "HEFT": 
            pi, where_scheduled = avg_graph.HEFT()
        elif heuristic == "HEFT-WM":
            pi, where_scheduled = avg_graph.HEFT(weighted=True)
        elif heuristic == "CPOP":
            pi, where_scheduled = avg_graph.CPOP()
        elif heuristic == "PEFT":
            pi, where_scheduled = avg_graph.PEFT()
        else:
            raise ValueError("Invalid heuristic type specified!")
        
        # Construct and return the schedule graph.
        S = self.graph.__class__()
        S.add_nodes_from(self.graph)
        S.add_edges_from(self.graph.edges)
        # Set the weights.
        for t in self.top_sort:
            w = where_scheduled[t]
            S.nodes[t]['weight'] = self.graph.nodes[t]['weight'][w] 
            for s in self.graph.successors(t):
                w1 = where_scheduled[s]
                S[t][s]['weight'] = self.graph[t][s]['weight'][(w, w1)] 
            # Add disjunctive edge if necessary.
            idx = list(r[0] for r in pi[w]).index(t)
            if idx > 0:
                d = pi[w][idx - 1][0]
                if not S.has_edge(d, t):
                    S.add_edge(d, t)
                    S[d][t]['weight'] = 0.0
        return SDAG(S) 
    
    def SDLS(self):
        """
        TODO: SDLS only computes an assignment, rather than a fullahead/strategy schedule. In previous implementation,
        used simple extension of Sculli's method with insertion based on mean values to obtain a task ordering. But original
        paper assumes no insertion, so maybe just go with that?        

        Returns
        -------
        None.

        """
        # Convert to stochastic averaged graph.
        S = self.get_averaged_graph(stochastic=True, avg_type="NORMAL")
        # Get upward ranks.
        U = S.sculli(direction="upward") # TODO: include task cost. Maybe create another function for all stochastic UR choices?
        # Compute deltas.
        # TODO: how to handle insertion - look at old code and original paper.
        # 
        # TODO: should this be an SDAG method?
        return
    def RobHEFT(self):
        return
    
# =============================================================================
# FUNCTIONS.
# =============================================================================

def stochastic_average(RVs, avg_type="NORMAL"):
    """Return an RV representing the average of a set of RVs (and possibly scalar zeros)."""
    
    # Check if all RVs are actually floats/0.0. Not ideal but necessary...
    if all((type(r) == float or type(r) == int)for r in RVs):
        return 0.0
    
    if avg_type in ["N", "NORMAL"]:
        L = len(RVs)
        mean = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.mu
        m = sum(mean(r) for r in RVs)      
        var = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.var
        v = sum(var(r) for r in RVs)
        return RV(m, v)/L
    # TODO: other types.
    return 0.0    
    
def clark(r1, r2, rho=0, minimization=False):
    """
    Returns a new RV representing the maximization of self and other whose mean and variance
    are computed using Clark's equations for the first two moments of the maximization of two normal RVs.
    TODO: minimization from one of Canon's papers, find source and cite.
    See:
    'The greatest of a finite set of random variables,'
    Charles E. Clark (1983).
    """
    a = sqrt(r1.var + r2.var - 2 * r1.sd * r2.sd * rho)     
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
         
        

        
                
