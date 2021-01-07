#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static scheduling for average costs only.
"""

import networkx as nx

class ADAG:
    """Represents a graph with average node and edge weights."""
    def __init__(self, graph):
        """Graph is a NetworkX digraph with {Processor ID : float} node and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def mean_task_cost(self, task, weighted=False):
        if not weighted:
            return sum(self.graph.nodes[task]['weight'].values())/len(self.graph.nodes[task]['weight'])
        s = sum(1/v for v in self.graph.nodes[task]['weight'].values())
        return len(self.graph.nodes[task]['weight']) / s 
    
    def mean_edge_cost(self, parent, child, weighted=False):
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
            ranks[t] = self.mean_task_cost(t, weighted=weighted)
            try:
                ranks[t] += max(self.mean_edge_cost(t, s, weighted=weighted) + ranks[s] for s in self.graph.successors(t))
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
        schedule, where = self.simulate_scheduling(priority_list=priority_list)
        # Compute makespan?
        # mkspan = max(schedule[w][-1][2] for w in self.graph.nodes[self.top_sort[0]]['weight'].keys())        
        return schedule, where

