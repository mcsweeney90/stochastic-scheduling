#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic scheduling simulator.
"""

import random
import networkx as nx
import numpy as np
import itertools as it
from math import sqrt, radians, tan
from psutil import virtual_memory
# from scipy.stats import norm # TODO: really slow but NormalDist only available for version >= 3.8. What to do on Matt's machine?
from statistics import NormalDist # TODO: see above, not available on Matt's machine.

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
            return random.gammavariate(alpha=(self.mu**2/self.var), beta=self.var/self.mu)      
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
        elif avg_type in ["SD", "sd"]: # TODO: for RobHEFT but necessary?
            return self.sd

class ADAG:
    """
    Represents a graph with average node and edge weights.
    TODO: rename this - is scalar graph so SDAG arguably makes more sense...
    """
    def __init__(self, graph):
        """Graph is a NetworkX digraph with {Processor ID : float} node and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def node_mean(self, task, weighted=False):
        """TODO."""
        if not weighted:
            return sum(self.graph.nodes[task]['weight'].values())/len(self.graph.nodes[task]['weight'])
        s = sum(1/v for v in self.graph.nodes[task]['weight'].values())
        return len(self.graph.nodes[task]['weight']) / s 
    
    def edge_mean(self, parent, child, weighted=False):
        """TODO: update, changes to edge weights."""
        if not weighted:
            return 2 * sum(self.graph[parent][child]['weight'].values()) / len(self.graph.nodes[parent]['weight'])**2
            # return sum(self.graph[parent][child]['weight'].values())/len(self.graph[parent][child]['weight']) 
        s1 = sum(1/v for v in self.graph.nodes[parent]['weight'].values())
        s2 = sum(1/v for v in self.graph.nodes[child]['weight'].values())
        cbar = 0.0
        for k, v in self.graph[parent][child]['weight'].items():
            t_w = self.graph.nodes[parent]['weight'][k[0]]
            c_w = self.graph.nodes[child]['weight'][k[1]]             
            cbar += v/(t_w * c_w) 
        cbar *= 2 # TODO: check.
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
    
    def comm_cost(self, parent, child, source, dest):
        """
        Get the communication/edge cost between parent and child when they are scheduled on source and dest (respectively).
        Assumes communication is symmetric.  
        """
        if source == dest:
            return 0.0
        elif source < dest:
            return self.graph[parent][child]['weight'][(source, dest)]
        else:
            return self.graph[parent][child]['weight'][(dest, source)] # symmetric.
    
    def simulate_scheduling(self, priority_list, policy="EFT"):
        """
        Simulates the scheduling of the tasks in priority_list.
        TODO: add a PEFT-like "makespan" policy? 
        """ 
        # Get list of workers and initialize schedule.
        workers = self.graph.nodes[self.top_sort[0]]['weight'].keys()
        schedule = {w : [] for w in workers}
        # Keep track of finish times and where tasks are scheduled.
        finish_times, where = {}, {}
        # Start the simulation.
        for task in priority_list:
            parents = list(self.graph.predecessors(task))
            worker_schedules = {}
            for w in workers:
                task_cost = self.graph.nodes[task]['weight'][w]
                # Find the data-ready time.       
                drt = 0.0 if not parents else max(finish_times[p] + self.comm_cost(p, task, where[p], w) for p in parents)
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
            min_worker = min(workers, key=lambda w:worker_schedules[w][1])
            where[task] = min_worker            
            st, ft, idx = worker_schedules[min_worker]
            finish_times[task] = ft  
            # Add to schedule.           
            if not schedule[min_worker] or idx < 0:             
                schedule[min_worker].append((task, st, ft))  
            else: 
                schedule[min_worker].insert(idx, (task, st, ft)) 
        return schedule, where    

class SDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is an NetworkX digraph with RV nodes and edge weights. Usually output by functions elsewhere..."""            
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def CPM(self, variance=False):
        """
        TODO: check this meshes with everything else. Maybe rewrite with lambda functions rather than the try statements.
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
        
    def longest_path(self, method="S", mc_dist="GAMMA", mc_samples=1000, full=False):
        """
        Evaluate the longest path through the entire DAG.
        TODO: Few possible issues with MC method:
            - Better way to determine memory limit.
            - Full/exit task only versions make the code unwieldy. Check everything works okay.
            - Worth ensuring positivity for normal and uniform? Do negative realizations ever occur?
        """
        
        if method in ["S", "s", "SCULLI", "sculli", "Sculli"]:
            L = {}
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
            return L[self.top_sort[-1]] if not full else L
        
        elif method in ["C", "c", "CORLCA", "CorLCA", "corLCA"]:
            L, V, dominant_ancestors = {}, {}, {}
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
            return L[self.top_sort[-1]] if not full else L
        
        elif method in ["MC", "mc", "MONTE CARLO", "Monte Carlo", "monte carlo"]:
            mem_limit = virtual_memory().available // 10 # Size of numpy random array ~ 8 * samples
            if self.size*mc_samples < mem_limit:        
                L = {}
                for t in self.top_sort:
                    m, s = self.graph.nodes[t]['weight'].mu, self.graph.nodes[t]['weight'].sd
                    if mc_dist in ["N", "NORMAL", "normal"]:  
                        # w = abs(np.random.normal(m, s, mc_samples))
                        w = np.random.normal(m, s, mc_samples)
                    elif mc_dist in ["G", "GAMMA", "gamma"]:
                        v = self.graph.nodes[t]['weight'].var
                        sh, sc = (m * m)/v, v/m
                        w = np.random.gamma(sh, sc, mc_samples)
                    elif mc_dist in ["U", "UNIFORM", "uniform"]:
                        u = sqrt(3) * s
                        w = np.random.uniform(-u + m, u + m, mc_samples) 
                        # w = abs(np.random.uniform(-u + m, u + m, mc_samples))
                    parents = list(self.graph.predecessors(t))
                    if not parents:
                        L[t] = w 
                        continue
                    pmatrix = []
                    for p in self.graph.predecessors(t):
                        try:
                            m, s = self.graph[p][t]['weight'].mu, self.graph[p][t]['weight'].sd
                            if mc_dist in ["N", "NORMAL", "normal"]: 
                                e = np.random.normal(m, s, mc_samples)
                                e = abs(np.random.normal(m, s, mc_samples))
                            elif mc_dist in ["G", "GAMMA", "gamma"]:
                                v = self.graph[p][t]['weight'].var
                                sh, sc = (m * m)/v, v/m
                                e = np.random.gamma(sh, sc, mc_samples)
                            elif mc_dist in ["U", "UNIFORM", "uniform"]:
                                u = sqrt(3) * s
                                e = np.random.uniform(-u + m, u + m, mc_samples)  
                                e = abs(np.random.uniform(-u + m, u + m, mc_samples))
                            pmatrix.append(np.add(L[p], e))
                        except AttributeError:
                            pmatrix.append(L[p])
                    st = np.amax(pmatrix, axis=0)
                    L[t] = np.add(w, st) 
                return L[self.top_sort[-1]] if not full else L  
            else:
                E = [] if not full else {}
                mx_samples = mem_limit//self.size
                runs = mc_samples//mx_samples
                extra = mc_samples % mx_samples
                for _ in range(runs):
                    if full:
                        L = self.longest_path(method="MC", mc_samples=mx_samples, mc_dist=mc_dist, full=True)
                        if len(E) == 0:
                            E = L
                        else:
                            for t in self.top_sort:
                                E[t] += L[t] # TODO: check if this throws an error because L[t] is an np array.
                    else:   
                        E += list(self.longest_path(method="MC", mc_samples=mx_samples, mc_dist=mc_dist))
                if full:
                    L = self.longest_path(method="MC", mc_samples=extra, mc_dist=mc_dist, full=True)
                    for t in self.top_sort:
                        E[t] += L[t] # TODO: check if this throws an error because L[t] is an np array.
                else:                    
                    E += list(self.longest_path(method="MC", mc_samples=extra, mc_dist=mc_dist))
                return E 
            
    def get_upward_ranks(self, method="S", mc_dist="NORMAL", mc_samples=1000):
        """
        
        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is "S".
        mc_dist : TYPE, optional
            DESCRIPTION. The default is "NORMAL".
        mc_samples : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.
        
        Not the most efficient way of doing this since the entire graph is copied, but the overhead is typically low compared
        to the cost of the longest path algorithms so this isn't a huge issue.
        Intended to be used for "averaged" graphs. 
        """    
        
        R = SDAG(self.graph.reverse())
        return R.longest_path(method=method, mc_dist=mc_dist, mc_samples=mc_samples, full=True)       
               
class TDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """
        Graph is a NetworkX digraph with {Processor ID : RV} node and edge weights. Usually output by functions elsewhere...
        TODO: create a "workers" attribute since often useful when weights have been set?
        """
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def set_weights(self, n_processors, cov, ccr=1.0, H=(1.0, 1.0, 1.0)):
        """
        Used for setting randomized weights for DAGs.
        H is a tuple that describes the (processor, task, communication) heterogeneity. All terms in [0, 2].
        CCR is only approximate but that's probably best here.
        cov is the mean coefficient of variation for ALL node and edge weight RVs. 
        """
        # Set the power of each processor.
        powers = {w : random.uniform(1-H[0]/2, 1+H[0]/2) for w in range(n_processors)}
        # Set the "bandwidth" of each link (to ensure some consistency in edge costs).
        B = {}
        for w in range(n_processors):
            for p in range(w + 1, n_processors): 
                B[(w, p)] = random.uniform(1-H[2]/2, 1+H[2]/2)
        
        # Set the task costs.
        exp_total_comp = 0.0
        for task in self.top_sort:
            self.graph.nodes[task]['weight'] = {}
            mean_task_cost = 100 * random.uniform(1-H[1]/2, 1+H[1]/2)  # Multiplied by 100 only to make task sizes more interesting...
            exp_total_comp += mean_task_cost
            for w in range(n_processors):
                mu = mean_task_cost * powers[w]
                sigma = random.uniform(0, 2*cov) * mu
                self.graph.nodes[task]['weight'][w] = RV(mu, sigma**2)
                
        # Set the edge costs.
        exp_edge_cost = (exp_total_comp/ccr)/self.graph.number_of_edges()   # Average of all edges means. 
        for task in self.top_sort:
            for child in self.graph.successors(task):
                self.graph[task][child]['weight'] = {}
                wbar = random.uniform(1-H[2]/2, 1+H[2]/2) * exp_edge_cost # Average mean of this particular edge.
                D = wbar * (n_processors/(n_processors-1))                
                for w in range(n_processors):
                    for p in range(w + 1, n_processors):
                        mu = D * B[(w, p)]
                        sigma = random.uniform(0, 2*cov) * mu
                        self.graph[task][child]['weight'][(w, p)] = RV(mu, sigma**2)
                        
    def minimal_serial_time(self, mc_samples=1000):
        """
        Extension of minimal serial time for scalar graphs. 
        Uses MC sampling to estimate the MST distribution, assuming that times are normally distributed (fairly
        reasonable as they are typically the sum of 100 or so task costs).  
        Assumes they are all independent (reasonable?)
        """
        workers = list(self.graph.nodes[self.top_sort[0]]['weight'])
        realizations = []
        for w in workers:
            dist = sum(self.graph.nodes[t]['weight'][w] for t in self.top_sort) 
            r = np.random.normal(dist.mu, dist.sd, mc_samples)
            realizations.append(r)
        return list(np.amin(realizations, axis=0))
    
    def get_scalar_graph(self, kind="AVERAGE", avg_type="MEAN", real_dist="GAMMA"):
        """Return an ADAG object..."""
        
        # Copy the topology.
        A = self.graph.__class__()
        A.add_nodes_from(self.graph)
        A.add_edges_from(self.graph.edges)
        
        # Set the weights.
        # edge_average = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.average(avg_type=avg_type) # TODO: needed (also real one)?
        for t in self.top_sort:
            # Set node weights.
            if kind in ["A", "AVERAGE", "average"]:
                A.nodes[t]['weight'] = {k : v.average(avg_type=avg_type) for k, v in self.graph.nodes[t]['weight'].items()}
            elif kind in ["R", "REALIZATION", "REAL", "realization", "real"]:
                A.nodes[t]['weight'] = {k : v.realize(dist=real_dist) for k, v in self.graph.nodes[t]['weight'].items()}
            # Set edge weights.
            for s in self.graph.successors(t):
                if kind in ["A", "AVERAGE", "average"]:                    
                    A[t][s]['weight'] = {k : v.average(avg_type=avg_type) for k, v in self.graph[t][s]['weight'].items()}
                elif kind in ["R", "REALIZATION", "REAL", "realization", "real"]:                
                    A[t][s]['weight'] = {k : v.realize(dist=real_dist) for k, v in self.graph[t][s]['weight'].items()} 
        # Return ADAG object.      
        return ADAG(A)
    
    def get_averaged_graph(self, avg_type="NORMAL"):
        """Return a graph with averaged weights..."""
        
        # Copy the topology.
        A = self.graph.__class__()
        A.add_nodes_from(self.graph)
        A.add_edges_from(self.graph.edges)
        
        # TODO: are these needed since changes?
        # mean = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.mu
        # var = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.var
                
        # Stochastic averages.
        if avg_type in ["N", "NORMAL", "CLT"]:
            L = len(self.graph.nodes[self.top_sort[0]]['weight'])
            L2 = L*L
            for t in self.top_sort:
                m = sum(r.mu for r in self.graph.nodes[t]['weight'].values()) 
                v = sum(r.var for r in self.graph.nodes[t]['weight'].values())
                A.nodes[t]['weight'] = RV(m, v)/L
                for s in self.graph.successors(t):
                    # m1 = 2*sum(mean(r) for r in self.graph[t][s]['weight'].values())
                    # v1 = 2*sum(var(r) for r in self.graph[t][s]['weight'].values())
                    m1 = 2*sum(r.mu for r in self.graph[t][s]['weight'].values())
                    v1 = 2*sum(r.var for r in self.graph[t][s]['weight'].values())
                    A[t][s]['weight'] = RV(m1, v1)/L2
            return SDAG(A)
        
        raise ValueError("Invalid stochastic average type!")        
                
    # def get_averaged_graph(self, stochastic=False, avg_type="MEAN"):
    #     """Return an equivalent graph with averaged weights."""
        
    #     # Copy the topology.
    #     A = self.graph.__class__()
    #     A.add_nodes_from(self.graph)
    #     A.add_edges_from(self.graph.edges)
        
    #     # Scalar averages.
    #     if not stochastic:
    #         for t in self.top_sort:
    #             A.nodes[t]['weight'] = {k : v.average(avg_type=avg_type) for k, v in self.graph.nodes[t]['weight'].items()}
    #             for s in self.graph.successors(t):
    #                 A[t][s]['weight'] = {}
    #                 average = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.average(avg_type=avg_type) 
    #                 # TODO: above may no longer be necessary.
    #                 A[t][s]['weight'] = {k : average(v) for k, v in self.graph[t][s]['weight'].items()} 
    #         return ADAG(A)
                
    #     # Stochastic averages. TODO: check the edge means since changes to set_weights.
    #     if avg_type in ["N", "NORMAL", "CLT"]:
    #         L = len(self.graph.nodes[self.top_sort[0]]['weight'])
    #         L2 = L*L
    #         mean = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.mu
    #         var = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.var
    #         for t in self.top_sort:
    #             m = sum(mean(r) for r in self.graph.nodes[t]['weight'].values()) # TODO: mean/var needed for nodes?
    #             v = sum(var(r) for r in self.graph.nodes[t]['weight'].values())
    #             A.nodes[t]['weight'] = RV(m, v)/L
    #             for s in self.graph.successors(t):
    #                 m1 = 2*sum(mean(r) for r in self.graph[t][s]['weight'].values())
    #                 v1 = 2*sum(var(r) for r in self.graph[t][s]['weight'].values())
    #                 A[t][s]['weight'] = RV(m1, v1)/L2
    #         return SDAG(A)
        
    #     raise ValueError("Invalid stochastic average type!")
    
    def comm_cost(self, parent, child, source, dest):
        if source == dest:
            return 0.0
        elif source < dest:
            return self.graph[parent][child]['weight'][(source, dest)]
        else:
            return self.graph[parent][child]['weight'][(dest, source)] # symmetric.
    
    def schedule_to_graph(self, schedule, where_scheduled=None):
        """
        Convert (fullahead) schedule to a "disjunctive" graph with stochastic weights whose longest path gives the makespan.
        
        Parameters
        ----------
        schedule : DICT
            {worker : [(task_id, est_start_time, est_finish_time), ...]}.
            Note est_start_time and est_finish_time may be scalar or RVs.
        where : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if where_scheduled is None:
            where_scheduled = {}
            for w, load in schedule.items():
                for t in list(s[0] for s in load):
                    where_scheduled[t] = w 
        
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
                S[t][s]['weight'] = self.comm_cost(t, s, w, w1)
            # Add disjunctive edge if necessary.
            idx = list(r[0] for r in schedule[w]).index(t)
            if idx > 0:
                d = schedule[w][idx - 1][0]
                if not S.has_edge(d, t):
                    S.add_edge(d, t)
                    S[d][t]['weight'] = 0.0
        return SDAG(S) 
    
# =============================================================================
# DETERMINISTIC HEURISTICS.
# =============================================================================

def HEFT(G, weighted=False):
    """
    HEFT scheduling heuristic.
    TODO: assumes G is an ADAG. Put in a check?
    """
    # Compute upward ranks.
    U = G.get_upward_ranks(weighted=weighted)
    # Sort into priority list.
    priority_list = list(sorted(U, key=U.get, reverse=True))
    # Get schedule.
    return G.simulate_scheduling(priority_list=priority_list) 

# =============================================================================
# STOCHASTIC HEURISTICS.
# =============================================================================

def SSTAR(S, det_heuristic=HEFT, avg_type="MEAN", weighted=False, scalar_graph=None):
    """
    TODO: don't like passing weighted as an argument - what to do in general?
    Converts a TDAG S to an "averaged" ADAG object based on avg_type, then applies det_heuristic to it. 
    When avg_type == "MEAN" and det_heuristic == HEFT, this is just HEFT applied to the stochastic graph.
    When avg_type == "MEAN" and det_heuristic == HEFT, this is the Stochastic HEFT (SHEFT) heuristic.        
    'A stochastic scheduling algorithm for precedence constrained tasks on Grid',
    Tang, Li, Liao, Fang, Wu (2011).
    However, this function can take any other deterministic heuristic instead.

    Parameters
    ----------
    S : TYPE
        DESCRIPTION.
    det_heuristic : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Convert to an "averaged" graph with scalar weights (if necessary).
    if scalar_graph is None:
        scalar_graph = S.get_scalar_graph(kind="A", avg_type=avg_type)
    # Apply the chosen heuristic.
    P, where = det_heuristic(scalar_graph, weighted=weighted) 
    # Convert fullahead schedule to its (stochastic) disjunctive graph and return.
    return S.schedule_to_graph(schedule=P, where_scheduled=where)
    

def SDLS(S, X=0.9, return_graph=True, insertion=None):
    """
    TODO: Insertion doesn't seem to make much of a difference
    TODO: Still may be too slow for large DAGs. Obviously copying graph etc is not optimal but those kind of things aren't the
    real bottlenecks.
    
    Returns
    -------
    None.

    """
    
    mean = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.mu # TODO: Hmmm... Only needed for insertion.
    
    # Get the list of workers - useful throughout.
    workers = list(S.graph.nodes[S.top_sort[0]]['weight'])
    
    # Convert to stochastic averaged graph.
    A = S.get_averaged_graph(avg_type="NORMAL")    
    # Get upward ranks via Sculli's method.
    B = A.get_upward_ranks(method="S") 
    
    # Compute the schedule.
    ready_tasks = [S.top_sort[0]]    # Assumes single entry task.   
    SDL, FT, where = {}, {}, {}     # FT and where make computing the SDL values easier. 
    schedule = {w : [] for w in workers}   # Loads are ordered
    while len(ready_tasks): 
        for task in ready_tasks:
            # Get the average task weight (used for computing the SDL values).
            avg_weight = A.graph.nodes[task]['weight']
            # Find all parents - useful for next part.
            parents = list(S.graph.predecessors(task)) 
            # Compute SDL value of the task on all workers.
            for w in workers:
                # Compute delta.
                delta = avg_weight - S.graph.nodes[task]['weight'][w]    
                
                # Compute earliest start time. 
                if not parents: # Single entry task.
                    EST, idx = 0.0, 0
                else:
                    p = parents[0]
                    # Compute DRT - start time without considering processor contention.
                    drt = FT[p] + S.comm_cost(p, task, where[p], w) 
                    for p in parents[1:]:
                        q = FT[p] + S.comm_cost(p, task, where[p], w) 
                        drt = clark(drt, q, rho=0)
                    # Find the earliest time task can be scheduled on the worker.
                    if not schedule[w]: # No tasks scheduled on worker. 
                        EST, idx = drt, 0
                    else:
                        if insertion is None:
                            EST, idx = clark(drt, schedule[w][-1][2], rho=0), len(schedule[w])
                        elif insertion in ["M", "MEAN"]: # mean instead of .mu - EST can be scalar...
                            task_cost = S.graph.nodes[task]['weight'][w].mu
                            found, prev_finish_time = False, 0.0
                            for i, t in enumerate(schedule[w]):
                                if mean(t[1]) < drt.mu:
                                    prev_finish_time = t[2]
                                    continue
                                try:
                                    poss_start_time = clark(prev_finish_time, drt) 
                                except AttributeError:
                                    poss_start_time = drt
                                poss_finish_time = poss_start_time + task_cost
                                if poss_finish_time.mu <= t[1].mu:
                                    found = True
                                    EST, idx = poss_start_time, i                           
                                    break
                                prev_finish_time = t[2]    
                            # No valid gap found.
                            if not found:
                                EST, idx = clark(drt, schedule[w][-1][2], rho=0), len(schedule[w])
                                                     
                # Compute SDL. (EST included as second argument to avoid having to recalculate it later but a bit ugly.)
                # SDL[(task, w)] = (B[task] - EST + delta, EST) if not insertion else (B[task] - EST + delta, EST, idx) 
                SDL[(task, w)] = (B[task] - EST + delta, EST, idx)
            
        # Select the maximum pair. 
        chosen_task, chosen_worker = max(it.product(ready_tasks, workers), 
                                          key=lambda pr : NormalDist(SDL[pr][0].mu, SDL[pr][0].sd).inv_cdf(X))
        # Comment above and uncomment below for Python versions < 3.8.
        # chosen_task, chosen_worker = max(it.product(ready_tasks, workers), key=lambda pr : norm.ppf(X, SDL[pr][0].mu, SDL[pr][0].sd))
                
        # Schedule the chosen task on the chosen worker. 
        where[chosen_task] = chosen_worker
        FT[chosen_task] = SDL[(chosen_task, chosen_worker)][1] + S.graph.nodes[chosen_task]['weight'][chosen_worker]
        schedule[chosen_worker].insert(SDL[(chosen_task, chosen_worker)][2], (chosen_task, SDL[(chosen_task, chosen_worker)][1], FT[chosen_task]))
        # if insertion:
        #     schedule[chosen_worker].insert(SDL[(chosen_task, chosen_worker)][2], (chosen_task, SDL[(chosen_task, chosen_worker)][1], FT[chosen_task])) 
        # else:
        #     schedule[chosen_worker].append((chosen_task, SDL[(chosen_task, chosen_worker)][1], FT[chosen_task])) 
        
        # Remove current task from ready set and add those now available for scheduling.
        ready_tasks.remove(chosen_task)
        for c in S.graph.successors(chosen_task):
            if all(p in where for p in S.graph.predecessors(c)):
                ready_tasks.append(c)   
                
    # If specified, convert schedule to graph and return it.
    if return_graph:
        return S.schedule_to_graph(schedule, where_scheduled=where)
    # Else return schedule only.
    return schedule  

def choose_worker(est_makespans, alpha=45):
    """
    Helper function for RobHEFT.

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    a : TYPE, optional
        DESCRIPTION. The default is 45.

    Returns
    -------
    None.

    """     
    # Filter the dominated workers.   
    sorted_workers = list(sorted(est_makespans, key=lambda p : est_makespans[p].mu)) # Ascending order of expected value.
    dominated = [False] * len(sorted_workers)           
    for i, w in enumerate(sorted_workers):
        if dominated[i]:
            continue
        for j, q in enumerate(sorted_workers[:i]):   
            if dominated[j]:
                continue
            if est_makespans[q].sd < est_makespans[w].sd:
                dominated[i] = True 
    nondominated = {w : est_makespans[w] for i, w in enumerate(sorted_workers) if not dominated[i]}
        
    # Find max mean and standard deviation for normalization.
    mxm = max(m.mu for m in nondominated.values())
    mxs = max(m.sd for m in nondominated.values())
    
    # Line segment runs from (0, 0) to (1, tan(alpha)) - but need to convert to radians first.
    line_end_pt = tan(radians(alpha)) 
    dist = lambda w : abs(line_end_pt * (nondominated[w].mu/mxm) - (nondominated[w].sd/mxs)) / sqrt(1 + line_end_pt**2)
    return min(nondominated, key=dist)    

def RobHEFT(S, alpha=45, mc_samples=None):
    """
    RobHEFT (HEFT with robustness) heuristic.
    'Evaluation and optimization of the robustness of DAG schedules in heterogeneous environments,'
    Canon and Jeannot (2010). 
    """
       
    workers = list(S.graph.nodes[S.top_sort[0]]['weight'])    # Useful.  
    n_workers = len(workers)
    
    # Compute priorities for all tasks.
    ranks = {}
    for task in reversed(S.top_sort):
        m = sum(r.mu for r in S.graph.nodes[task]['weight'].values())/n_workers
        v = sum(r.var for r in S.graph.nodes[task]['weight'].values())/(n_workers*n_workers)            
        ranks[task] = RV(m, v)
        children = S.graph.successors(task)
        ST = RV(0.0, 0.0)
        for child in children:
            m1 = 2*sum(r.mu for r in S.graph[task][child]['weight'].values())/(n_workers*n_workers) 
            v1 = 2*sum(r.var for r in S.graph[task][child]['weight'].values())/(n_workers**4)
            child_st = RV(m1, v1) + ranks[child]
            if child_st.mu > ST.mu:
                ST = child_st
        ranks[task] += ST
    # Get maximums for later normalization.
    mx_mu = ranks[S.top_sort[0]].mu
    mx_sd = max(ranks[t].sd for t in S.top_sort)    
           
    # Create the schedule graph.
    # TODO: remove and just use correlation tree? 
    G = S.graph.__class__()       
    
    # Simulate and find schedule.  
    where, last = {}, {} # Helpers.
    ready_tasks = [S.top_sort[0]] 
    while ready_tasks:
        task = max(ready_tasks, key=lambda t : alpha*(ranks[t].mu/mx_mu) + (90-alpha)*(ranks[t].sd/mx_sd))
        G.add_node(task)
        parents = list(S.graph.predecessors(task))
        for p in parents:
            G.add_edge(p, task)
        worker_makespans = {}
        for w in workers:
            
            # Set schedule node weights. 
            G.nodes[task]['weight'] = S.graph.nodes[task]['weight'][w]
            # Same for edges.
            for p in parents:
                G[p][task]['weight'] = S.comm_cost(p, task, where[p], w)
                
            # Add the transitive edge if necessary. TODO: insertion?
            remove = False
            try:
                L = last[w]
                if not G.has_edge(L, task):
                    G.add_edge(L, task)
                    G[L][task]['weight'] = 0.0
                    remove = True
            except KeyError:
                pass
            
            # Add artificial exit node if necessary. TODO: don't like this at all. And very slow.
            exit_tasks = [t for t in G if not len(list(G.successors(t)))]
            if len(exit_tasks) > 1:
                G.add_node("X")
                G.nodes["X"]['weight'] = RV(0.0, 0.0) # don't like.
                for e in exit_tasks:
                    G.add_edge(e, "X")
                    G[e]["X"]['weight'] = 0.0 
                    
            # Compute longest path using either CorLCA or MC.
            P = SDAG(G)
            if mc_samples is not None:
                worker_dist = P.longest_path(method="MC", mc_dist="N", mc_samples=mc_samples) # TODO: assume normal costs?
                m = sum(worker_dist)/len(worker_dist)
                v = np.var(worker_dist)
                worker_makespans[w] = RV(m, v)
            else:
                worker_makespans[w] = P.longest_path(method="C")
            
            # Clean up - remove edge etc. TODO: need to set node weight to zero?
            if remove:
                G.remove_edge(L, task)
            if len(exit_tasks) > 1:
                G.remove_node("X")
                
        # Select the "best" worker according to the aggregation/angle procedure.
        chosen_worker = choose_worker(worker_makespans, alpha)
        # "Schedule" the task on chosen worker.
        where[task] = chosen_worker
        G.nodes[task]['weight'] = S.graph.nodes[task]['weight'][chosen_worker]
        # Same for edges.
        for p in parents:
            G[p][task]['weight'] = S.comm_cost(p, task, where[p], chosen_worker)
        try:
            L = last[chosen_worker]
            if not G.has_edge(L, task):
                G.add_edge(L, task)
                G[L][task]['weight'] = 0.0
        except KeyError:
            pass
        last[chosen_worker] = task
        
        # Remove current task from ready set and add those now available for scheduling.
        ready_tasks.remove(task)
        for c in S.graph.successors(task):
            if all(p in where for p in S.graph.predecessors(c)):
                ready_tasks.append(c) 
   
    return SDAG(G)

# def GGB(G, priorities=None, policy="EFT", insertion=False, eval_method="C"):
#     """TODO. Schedule the list in order according to the policy."""
    
#     workers = list(G.graph.nodes[G.top_sort[0]]['weight'])
    
#     # Create and grow the schedule graph.
#     S = G.graph.__class__() 
    
#     if priorities is not None:
#         prios = priorities()
#     else:
#         prios = 0         
    
#     where, loads, last = {}, {}, {}
#     for task in task_list:
#         S.add_node(task)
#         parents = list(self.graph.predecessors(task)) 
#         for p in parents:
#             S.add_edge(p, task)
#         worker_makespans = {}
#         for w in workers:
#             # Set schedule node weights. TODO: add node instead?
#             S.nodes[task]['weight'] = self.graph.nodes[task]['weight'][w]
#             # Same for edges.
#             for p in parents:
#                 S[p][task]['weight'] = self.comm_costs(p, task, where[p], w)
            
#             # Transitive edges. TODO: insertion.
#             remove = False
#             try:
#                 L = last[w]
#                 if not S.has_edge(L, task):
#                     S.add_edge(L, task)
#                     S[L][task]['weight'] = 0.0
#                     remove = True
#             except KeyError:
#                 pass
            
#             # Possibly need to add single exit task.
#             exit_tasks = [t for t in S if not len(list(S.successors(t)))]
#             if len(exit_tasks) > 1:
#                 S.add_node("X")
#                 S.nodes["X"]['weight'] = RV(0.0, 0.0) # don't like.
#                 for e in exit_tasks:
#                     S.add_edge(e, "X")
#                     S[e]["X"]['weight'] = 0.0 
                
#             # Evaluate the makespan.
#             P = SDAG(S)
#             worker_makespans[w] = P.longest_path(method=eval_method)
            
#             # Clean up - remove edge etc. TODO: need to set node weight to zero?
#             if remove:
#                 S.remove_edge(L, task)
#             if len(exit_tasks) > 1:
#                 S.remove_node("X")
                
#         # print(worker_makespans)
#         # Select the "best" worker according to the aggregation/angle procedure.
#         # chosen_worker = closest_worker(worker_makespans, a=a)
#         chosen_worker = min(workers, key=lambda w:worker_makespans[w].mu)
#         # print(chosen_worker)
#         # "Schedule" the task on chosen worker.
#         where[task] = chosen_worker
#         S.nodes[task]['weight'] = self.graph.nodes[task]['weight'][chosen_worker]
#         # Same for edges.
#         for p in parents:
#             S[p][task]['weight'] = self.graph[p][task]['weight'][(where[p], chosen_worker)]
#         try:
#             L = last[chosen_worker]
#             if not S.has_edge(L, task):
#                 S.add_edge(L, task)
#                 S[L][task]['weight'] = 0.0
#         except KeyError:
#             pass
#         last[chosen_worker] = task
            
#     return SDAG(S) 

def MCS(S, production_heuristic=HEFT, production_steps=100, threshold=0.2, dist="N", samples=1000):
    """ 
    Monte Carlo Scheduling (MCS).
    'Stochastic DAG scheduling using a Monte Carlo approach,'
    Zheng and Sakellariou (2013).
    TODO: seems to work but almost never overrules HEFT - bug somewhere or just parameters/graph instances?
    """
    
    # Create schedule list.
    L = []
    
    # Get the standard HEFT schedule.
    avg_graph = S.get_scalar_graph(kind="A", avg_type="MEAN")
    mean_static_schedule, where = production_heuristic(avg_graph) 
    omega_mean = S.schedule_to_graph(schedule=mean_static_schedule, where_scheduled=where)
    qualify_check = omega_mean.CPM()[S.top_sort[-1]] * (1 + threshold) # TODO. Think this works but verify. Dynamically updated in 
    L.append(omega_mean)
    
    # Production step.
    for _ in range(production_steps):
        G = S.get_scalar_graph(kind="R", real_dist=dist) 
        static_schedule, where = production_heuristic(G)
        omega = S.schedule_to_graph(schedule=static_schedule, where_scheduled=where)
        # if all(omega.graph.edges != pi.graph.edges for pi in L): # TODO: doesn't work, override comparator for SDAGs?
        #     continue
        # Calculate CPM longest path of omega and compare with average.
        if omega.CPM()[S.top_sort[-1]] < qualify_check: 
            L.append(omega)  
    print(len(L))
    # Evaluate the makespan distributions of the candidiate schedules.
    makespans = {pi : pi.longest_path(method="MC", mc_dist=dist, mc_samples=samples) for pi in L}  
    # print(list(sum(makespans[pi])/len(makespans[pi]) for pi in L))
    return min(L, key=lambda pi : sum(makespans[pi])/len(makespans[pi]))    
    

    
# =============================================================================
# GENERAL FUNCTIONS.
# =============================================================================   
    
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
    cdf = NormalDist().cdf(b)   #norm.cdf(b)
    mcdf = 1 - cdf 
    pdf = NormalDist().pdf(b)   #norm.pdf(b)   
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
        
        

        
                
