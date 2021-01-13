#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic scheduling simulator.
"""

import random
import networkx as nx
import numpy as np
import itertools as it
from math import sqrt
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
        elif avg_type in ["SD", "sd"]: # TODO: for RobHEFT but necessary?
            return self.sd

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
    def PEFT(self):
        return

class SDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is an NetworkX digraph with RV nodes and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def longest_path(self, method="S", mc_dist="GAMMA", mc_samples=1000, full=False):
        """
        Evaluate the longest path through the entire DAG.
        MC:
        TODO: Better way to determine memory limit?
        TODO: no check if positive for normal and uniform!
        TODO: recursion doesn't work if full == True.
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
                        w = np.random.normal(m, s, mc_samples)
                    elif mc_dist in ["G", "GAMMA", "gamma"]:
                        v = self.graph.nodes[t]['weight'].var
                        sh, sc = (m * m)/v, v/m
                        w = np.random.gamma(sh, sc, mc_samples)
                    elif mc_dist in ["U", "UNIFORM", "uniform"]:
                        u = sqrt(3) * s
                        w = np.random.uniform(-u + m, u + m, mc_samples) 
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
                            elif mc_dist in ["G", "GAMMA", "gamma"]:
                                v = self.graph[p][t]['weight'].var
                                sh, sc = (m * m)/v, v/m
                                e = np.random.gamma(sh, sc, mc_samples)
                            elif mc_dist in ["U", "UNIFORM", "uniform"]:
                                u = sqrt(3) * s
                                e = np.random.uniform(-u + m, u + m, mc_samples)  
                            pmatrix.append(np.add(L[p], e))
                        except AttributeError:
                            pmatrix.append(L[p])
                    st = np.amax(pmatrix, axis=0)
                    L[t] = np.add(w, st)
                return L[self.top_sort[-1]] if not full else L  # TODO: recursion doesn't work with full == True.
            else:
                E = []
                mx_samples = mem_limit//self.size
                runs = mc_samples//mx_samples
                extra = mc_samples % mx_samples
                for _ in range(runs):
                    E += list(self.longest_path(method="MC", mc_samples=mx_samples, mc_dist=mc_dist))
                E += list(self.longest_path(method="MC", mc_samples=extra, mc_dist=mc_dist))
                return E # TODO: Doesn't work with full == True.
            
    def get_upward_ranks(self, method="S", mc_dist="GAMMA", mc_samples=1000):
        """
        
        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is "S".
        mc_dist : TYPE, optional
            DESCRIPTION. The default is "GAMMA".
        mc_samples : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.
        
        Not the most efficient way of doing this since the entire graph is copied, but the overhead is typically low compared
        to the cost of the longest path algorithms so this isn't a huge issue.
        """    
        
        R = SDAG(self.graph.reverse())
        return R.longest_path(method=method, mc_dist=mc_dist, mc_samples=mc_samples, full=True)       
               
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
    
    def schedule_to_graph(self, schedule, where_scheduled=None):
        """
        Convert schedule into a graph with stochastic weights whose longest path gives the makespan.
        
        Parameters
        ----------
        schedule : TYPE
            DESCRIPTION.
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
                S[t][s]['weight'] = self.graph[t][s]['weight'][(w, w1)] 
            # Add disjunctive edge if necessary.
            idx = list(r[0] for r in schedule[w]).index(t)
            if idx > 0:
                d = schedule[w][idx - 1][0]
                if not S.has_edge(d, t):
                    S.add_edge(d, t)
                    S[d][t]['weight'] = 0.0
        return SDAG(S) 
    
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
            pi, where = avg_graph.HEFT()
        elif heuristic == "HEFT-WM":
            pi, where = avg_graph.HEFT(weighted=True)
        elif heuristic == "PEFT":
            pi, where = avg_graph.PEFT()
        else:
            raise ValueError("Invalid heuristic type specified!")
        
        return self.schedule_to_graph(schedule=pi, where_scheduled=where)
    
    def SDLS(self, X=0.9, return_graph=True):
        """
        TODO: Insertion.
        TODO: Still may be too slow for large DAGs. Obviously copying graph etc is not optimal but those kind of things aren't the
        real bottlenecks.

        Returns
        -------
        None.

        """
        # Convert to stochastic averaged graph.
        S = self.get_averaged_graph(stochastic=True, avg_type="NORMAL")
        
        # Get upward ranks.
        B = S.get_upward_ranks(method="S") # Upward ranks computed via Sculli's method.
        
        # Compute the schedule.
        workers = list(self.graph.nodes[self.top_sort[0]]['weight'].keys())    # Useful.
        ready_tasks = [self.top_sort[0]]    # Assumes single entry task.   
        SDL, FT, where = {}, {}, {}
        schedule = {w : [] for w in workers}   
        while len(ready_tasks): 
            for task in ready_tasks:
                avg_weight = stochastic_average(self.graph.nodes[task]['weight'].values(), avg_type="NORMAL") # TODO: check. 
                parents = list(self.graph.predecessors(task))
                for w in workers:
                    # Compute delta.
                    delta = avg_weight - self.graph.nodes[task]['weight'][w]    
                    
                    # Compute earliest start time. 
                    if not parents: # Single entry task.
                        EST = 0.0
                    else:
                        p = parents[0]
                        # Compute DRT - start time without considering processor contention.
                        drt = FT[p] + self.graph[p][task]['weight'][(where[p], w)]
                        for p in parents[1:]:
                            q = FT[p] + self.graph[p][task]['weight'][(where[p], w)]
                            drt = clark(drt, q, rho=0)
                        # Find the earliest time task can be scheduled on the worker.
                        if not schedule[w]: # No tasks scheduled on worker. TODO: insertion.
                            EST = drt
                        else:
                            EST = clark(drt, schedule[w][-1][2], rho=0)
                                                         
                    # Compute SDL.
                    SDL[(task, w)] = (B[task] - EST + delta, EST)    # Returns EST to prevent having to recalculate it later but a bit ugly.
                
            # Select the maximum pair. TODO: bit much for a one liner? Make sure it works...
            # chosen_task, chosen_worker = max(it.product(ready_tasks, workers), key=lambda pr : norm.ppf(X, SDL[pr][0].mu, SDL[pr][0].sd))
            chosen_task, chosen_worker = max(it.product(ready_tasks, workers), 
                                              key=lambda pr : NormalDist(SDL[pr][0].mu, SDL[pr][0].sd).inv_cdf(X))
                    
            # Schedule the chosen task on the chosen worker. 
            where[chosen_task] = chosen_worker
            FT[chosen_task] = SDL[(chosen_task, chosen_worker)][1] + self.graph.nodes[chosen_task]['weight'][chosen_worker]
            schedule[chosen_worker].append((chosen_task, SDL[(chosen_task, chosen_worker)][1], FT[chosen_task])) 
            
            ready_tasks.remove(chosen_task)
            for c in self.graph.successors(chosen_task):
                if all(p in where for p in self.graph.predecessors(c)):
                    ready_tasks.append(c)   
                    
        # If necessary, convert schedule to graph.
        if return_graph:
            return self.schedule_to_graph(schedule, where_scheduled=where)
        return schedule    
    
    def RobHEFT(self, a=45):
        """
        RobHEFT (HEFT with robustness) heuristic.
        'Evaluation and optimization of the robustness of DAG schedules in heterogeneous environments,'
        Canon and Jeannot (2010). 
        """
        
        # Get priority list of tasks.
        A1 = self.get_averaged_graph(avg_type="MEAN")
        A2 = self.get_averaged_graph(avg_type="SD")
        UM = A1.get_upward_ranks()
        mmx = max(UM.values()) # 
        US = A2.get_upward_ranks()
        smx = max(US.values())
        U = {t : a * (UM[t]/mmx) + (90 - a) * (US[t]/smx) for t in self.top_sort}
        priority_list = list(sorted(self.top_sort, key=U.get)) # TODO: ascending or descending?
        
        pi = nx.DiGraph()   # TODO: copy here and set all weights to 0.0?
        workers = list(self.graph.nodes[self.top_sort[0]]['weight'].keys())    # Useful.   
        where = {}
        last = {w : 0 for w in workers}
        for task in priority_list:
            parents = list(self.graph.predecessors(task))
            for w in workers:
                # Set schedule node weights.
                pi.graph.nodes[task]['weight'] = self.graph.nodes[task]['weight'][w]
                # Same for edges.
                for p in parents:
                    pi.graph[p][task]['weight'] = self.graph[p][task]['weight'][(where[p], w)]
                # TODO: Add the transitive edge if necessary.
                x = 0
                # TODO: Compute longest path using either CorLCA or MC.
                y = 0
                # Add to e.g. worker_finish_times.
                # Clean up - remove edge etc.
            # Select the "best" worker according to the aggregation/angle procedure.
                
                
            
            
            
            
        
        
        
        return
    
# =============================================================================
# FUNCTIONS.
# =============================================================================

def stochastic_average(RVs, avg_type="NORMAL"):
    """Return an RV representing the average of a set of RVs (and possibly scalar zeros)."""
    
    # Check if all RVs are actually floats/0.0. Not ideal but necessary...
    if all((type(r) == float or type(r) == int) for r in RVs):
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
    TODO: this is slow because of the calls to norm.pdf and norm.cdf. Is there any way to vectorize/optimize another way?
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
         
        

        
                
