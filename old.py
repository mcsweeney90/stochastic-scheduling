#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: the following is all copied and pasted from old stochastic scheudling repo so very out-of-date.
Will all be updated (after Christmas!).
"""


import dill
import numpy as np
import networkx as nx
from copy import deepcopy
from networkx.drawing.nx_agraph import to_agraph
from networkx.drawing.nx_pydot import read_dot
from collections import defaultdict
from scipy.stats import norm

class RV:
    """
    Random variable class.
    Implicitly assumed to be Gaussian but may do something different in future.
    TODO: should really add some checks for e.g., trying to multiply two RVs.
    """
    def __init__(self, mu=0, var=0, realization=None, ID=None):
        self.mu = mu
        self.var = var
        self.realization = realization
        self.ID = ID
    def __repr__(self):
        return "N({}, {})".format(self.mu, self.var)
    def __add__(self, other): # Costs are typically independent so don't think there's any reason to ever consider correlation here.
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu + other, self.var)
        return RV(self.mu + other.mu, self.var + other.var) 
    __radd__ = __add__ # Other way around...
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu - other, self.var)
        return RV(self.mu - other.mu, self.var + other.var)
    __rsub__ = __sub__ # Other way around...
    def __mul__(self, c):
        return RV(c * self.mu, c * c * self.var)
    __rmul__ = __mul__ # Other way around...
    def __truediv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rtruediv__ = __truediv__ # Other way around...
    def __floordiv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rfloordiv__ = __floordiv__ # Other way around...
    
    def clark_max(self, other, rho=0):
        if self.var == 0 and other.var == 0:
            return RV(max(self.mu, other.mu), 0)
        a = np.sqrt(self.var + other.var - 2 * np.sqrt(self.var) * np.sqrt(other.var) * rho)
#        print("a = {}".format(a))
        alpha = (self.mu - other.mu) / a
#        print("alpha = {}".format(alpha))
            
        Phi_alpha = norm.cdf(alpha)
#        print("Phi_alpha = {}".format(Phi_alpha))
        Phi_minus = norm.cdf(-alpha)
#        print("Phi_minus = {}".format(Phi_minus))
        Psi_alpha = norm.pdf(alpha) 
#        print("Psi_alpha = {}".format(Psi_alpha))
        
        mu = self.mu * Phi_alpha + other.mu * Phi_minus + a * Psi_alpha      
        var = (self.mu**2 + self.var) * Phi_alpha
        var += (other.mu**2 + other.var) * Phi_minus
        var += (self.mu + other.mu) * a * Psi_alpha
        var -= mu**2         
        return RV(mu, var)
    
    def clark_min(self, other, rho=0):
        if self.var == 0 and other.var == 0:
            return RV(min(self.mu, other.mu), 0)
        a = np.sqrt(self.var + other.var - 2 * np.sqrt(self.var) * np.sqrt(other.var) * rho)
#        print("a = {}".format(a))
        alpha = (self.mu - other.mu) / a
#        print("alpha = {}".format(alpha))
            
        Phi_alpha = norm.cdf(alpha)
#        print("Phi_alpha = {}".format(Phi_alpha))
        Phi_minus = norm.cdf(-alpha)
#        print("Phi_minus = {}".format(Phi_minus))
        Psi_alpha = norm.pdf(alpha) 
#        print("Psi_alpha = {}".format(Psi_alpha))
        
        mu = self.mu * Phi_minus + other.mu * Phi_alpha - a * Psi_alpha      
        var = (self.mu**2 + self.var) * Phi_minus
        var += (other.mu**2 + other.var) * Phi_alpha
        var -= (self.mu + other.mu) * a * Psi_alpha
        var -= mu**2         
        return RV(mu, var)
    
    def make_static(self):
        self.realization = self.mu
    def is_static(self):
        return (self.realization == self.mu)
    def reset(self):
        self.mu, self.var, self.realization = 0, 0, None
    def realize(self, sample=None, dist=False, alpha=1.0): # TODO.
        if sample is not None:                
            self.realization = np.random.choice(sample)            
        elif dist:
            self.realization = np.random.normal(self.mu, np.sqrt(self.var))              
        else:
            self.realization = np.random.uniform(alpha * self.mu, (2 - alpha) * self.mu)        
    def approx_weight(self):
        if self.var > self.mu**2:
            return self.mu + self.mu / np.sqrt(self.var)
        else:
            return self.mu + np.sqrt(self.var)

class Task:
    """
    Represent tasks.
    """         
    def __init__(self, task_type=None):
        """
        Create Task object.
        
        Parameters
        ------------------------
        task_type - None/string
        String identifying the name of the task, e.g., "GEMM".
        
        Attributes
        ------------------------
        type - None/string
        Initialized to task_type.
        
        ID - int
        Identification number of the Task in its DAG.
        
        entry - bool
        True if Task has no predecessors, False otherwise.
        
        exit - bool
        True if Task has no successors, False otherwise.
        
        The following 4 attributes are usually set after initialization by functions which
        take a Node object as a target platform.
        
        CPU_cost - int/float
        The Task's execution time on CPU Workers. 
        
        GPU_cost - int/float
        The Task's execution time on GPU Workers. 
        
        acceleration_ratio - int/float
        The ratio of the Task's execution time on CPU and GPU Workers. 
        
        comm_costs - defaultdict(dict)
        Nested dict {string identifying source and target processor types : {child ID : cost}}
        e.g., self.comm_costs["CG"][5] = 10 means that the communication cost between the Task
        and the child task with ID 5 is 10 when Task is scheduled on a CPU Worker and the child 
        is scheduled on a GPU Worker.
        
        The following 4 attributes are set once the task has actually been scheduled.
        
        AST - int/float
        The actual start time of the Task.
        
        AFT- int/float
        The actual finish time of the Task.
        
        scheduled - bool
        True if Task has been scheduled on a Worker, False otherwise.
        
        where_scheduled - None/int
        The numerical ID of the Worker that the Task has been scheduled on. Often useful.
        
        Comments
        ------------------------
        1. It would perhaps be more useful in general to take all attributes as parameters since this
           is more flexible but as we rarely work at the level of individual Tasks this isn't necessary
           for our purposes.        
        """  
        
        self.type = task_type 
        self.ID = None    
        self.entry = False  
        self.exit = False   
        
        # Expected computation and communication times, set by methods that take a platform as input.
        self.CPU_cost = RV()  
        self.GPU_cost = RV()   
        self.acceleration_ratio = {"expected" : 0.0, "actual" : 0.0}
        
        self.comm_costs = defaultdict(dict) # Nested dict, e.g., comm_costs["CC"][child.ID] == CPU-CPU cost from parent to child.
        self.comm_costs["CC"], self.comm_costs["CG"], self.comm_costs["GC"], self.comm_costs["GG"] = {}, {}, {}, {}  
        
        # Actual computation and communication costs.
        self.is_static = False # Often useful.
        self.actual_costs_set = False # Useful sometimes as a quick check.
                
        # Set once task has been scheduled.
        self.ST = RV()  # Start time.
        self.FT = RV()  # Finish time.
        self.scheduled = False   # True if task has been scheduled somewhere.
        self.where_scheduled = None   # ID of processor task is scheduled on, sometimes useful.               
    
    def reset(self):
        """Resets some attributes to defaults so execution of the task can be simulated again."""
        self.ST.reset()
        self.FT.reset() 
        self.scheduled = False
        self.where_scheduled = None       
        
    def average_comp_cost(self, platform, avg_type="HEFT", cost_type="stochastic"):
        """
        Compute the "average" computation time of the Task. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
                
        avg_type - string
        How the average should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", always use largest possible computation cost.
            - "simple worst", always use largest possible computation cost.
            - "best", always use smallest possible computation cost.
            - "simple best", always use smallest possible computation cost.
            - "HEFT-WM", compute mean over all processors, weighted by acceleration ratio.
            - "PS", processor speedup. Cost = max(CPU cost, GPU cost) / min(CPU cost, GPU cost).
            - "D", difference. Cost = max(CPU cost, GPU cost) - min(CPU cost, GPU cost).
            - "SFB". Cost = ( max(CPU cost, GPU cost) - min(CPU cost, GPU cost) ) / ( max(CPU cost, GPU cost) / min(CPU cost, GPU cost) ). 
                                         
        Returns
        ------------------------
        float 
        The approximate computation cost of the Task. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou (2003). 
        2. "PS", "D" and "SFB" are from Shetti, Fahmy and Bretschneider (2013).
        3. Average is perhaps the wrong word for some of these but it's useful to include them in this function.
        """
        
        if cost_type == "var":
            C, G = self.CPU_cost.mu, self.GPU_cost.mu
        elif cost_type == "std":
            C, G = np.sqrt(self.CPU_cost.var), np.sqrt(self.GPU_cost.var)
        elif cost_type == "actual":
            C, G = self.CPU_cost.actual, self.GPU_cost.actual
        elif cost_type == "expected":
            C, G = self.CPU_cost.mu, self.GPU_cost.mu
        elif cost_type == "SHEFT":
            C = self.CPU_cost.approx_weight()
            G = self.GPU_cost.approx_weight()                
        else:
            C, G = self.CPU_cost, self.GPU_cost
        
        if avg_type == "HEFT" or avg_type == "SHEFT" or avg_type == "mean":
            return (C * platform.n_CPUs + G * platform.n_GPUs) / platform.n_workers         
        
        elif avg_type == "worst" or avg_type == "W" or avg_type == "simple worst" or avg_type == "SW":
            if isinstance(C, float) or isinstance(C, int):
                return max(C, G)
            return C.clark_max(G)
        
        elif avg_type == "best" or avg_type == "B" or avg_type == "simple best" or avg_type == "sb":
            if isinstance(C, float) or isinstance(C, int):
                return min(C, G)
            return C.clark_min(G)  
        
        elif avg_type == "HEFT-WM" or avg_type == "WM":
            r = self.acceleration_ratio["expected"]
            return (C * platform.n_CPUs + r * G * platform.n_GPUs) / (platform.n_CPUs + r * platform.n_GPUs)  
    
        elif avg_type == "PS" or avg_type == "ps": # "PS" == "processor speedup".
            if isinstance(C, float) or isinstance(C, int):                
                fastest, slowest = min(C, G), max(C, G)
                if not fastest:
                    return 0
                return slowest / fastest   
            raise ValueError("Haven't decided how to extend avg_type == {} to the stochastic case yet! Watch this space...".format(avg_type)) 
        
        elif avg_type == "diff" or avg_type == "D":
            if isinstance(C, float) or isinstance(C, int): 
                fastest, slowest = min(C, G), max(C, G)
                return slowest - fastest
            raise ValueError("Haven't decided how to extend avg_type == {} to the stochastic case yet! Watch this space...".format(avg_type)) 
        
        elif avg_type == "sfb" or avg_type == "SFB":
            if isinstance(C, float) or isinstance(C, int): 
                fastest, slowest = min(C, G), max(C, G)
                if not fastest:
                    return 0
                return (slowest - fastest) / (slowest / fastest)  
            raise ValueError("Haven't decided how to extend avg_type == {} to the stochastic case yet! Watch this space...".format(avg_type))
                    
        raise ValueError('No avg_type, e.g., "mean" or "median", specified for average_comp_cost!')   

    def ready_to_schedule(self, dag):
        """
        Determine if Task is ready to schedule - i.e., all precedence constraints have been 
        satisfied or it is an entry task.
        
        Parameters
        ------------------------
        dag - DAG object
        The DAG to which the Task belongs.                
                                         
        Returns
        ------------------------
        bool
        True if Task can be scheduled, False otherwise.         
        
        Notes
        ------------------------
        1. Returns False if Task has already been scheduled.
        """
        
        if self.scheduled:
            return False
        if self.entry:  
            return True
        for p in dag.DAG.predecessors(self):
            if not p.scheduled:
                return False
        return True
    
    def make_static(self):
        """Convert to a static Task by setting all actual costs to their expected counterparts."""
        self.CPU_cost.make_static() 
        self.GPU_cost.make_static() 
        self.acceleration_ratio["actual"] = self.acceleration_ratio["expected"]
        for comm_type in ["CC", "CG", "GC", "GG"]:
            for child in self.comm_costs[comm_type].values():
                child.make_static()
        if not self.actual_costs_set:
            self.actual_costs_set = True  
        self.is_static = True

class DAG:
    """
    Represents a task DAG.   
    """
    def __init__(self, app="Random"): 
        """
        The DAG is a collection of Tasks with a topology defined by a Networkx DiGraph object.        
        
        Parameters
        ------------------------
        app - string
        The name of application the DAG represents, e.g., "Cholesky".
        
        Attributes
        ------------------------
        app - string
        Ditto above.
        
        DAG - DiGraph from Networkx module
        Represents the topology of the DAG.
        
        num_tasks - int
        The number of tasks in the DAG.
        
        The following attributes summarize topological information and are usually set
        by compute_topological_info when necessary.
        
        max_task_predecessors - None/int
        The maximum number of predecessors possessed by any task in the DAG.    
        
        avg_task_predecessors - None/int
        The average number of predecessors possessed by all tasks in the DAG.
               
        num_edges - None/int
        The number of edges in the DAG. 
        
        edge_density - None/float
        The ratio of the number of edges in the DAG to the maximum possible for a DAG with the same
        number of tasks. 
        
        CCR - Nested dict {string : {string : float}}
        Summarizes the computation-to-communication ratio (CCR) values for different platforms 
        in the form {platform name : {"expected" : expected CCR, "actual" : actual CCR}, ...}.          
        
        all_actual_costs_set - bool
        Indicates if all actual cost attributes have been set for all Tasks in the DAG.
        
        Comments
        ------------------------
        1. It seems a little strange to make the CCR a dict but it avoided having to compute it over 
           and over again for the same platforms in some scripts.
        """  
        
        self.app = app 
        self.DAG = nx.DiGraph()
        self.num_tasks = 0
        self.max_task_predecessors = None
        self.avg_task_predecessors = None
        self.num_edges = None
        self.edge_density = None 
        self.CCR = defaultdict(lambda : defaultdict(float)) # TODO: don't like this, make it an RV?
        self.all_actual_costs_set = False

    def compute_topological_info(self):
        """
        Compute information about the DAG's topology and set the corresponding attributes.                
             
        Notes
        ------------------------
        1. The maximum number of edges for a DAG with n tasks is 1/2 * n * (n - 1).
           This can be proven e.g., by considering each vertex in turn and determining
           the maximum number of possible new edges.
        """  
        
        if self.max_task_predecessors is None and self.avg_task_predecessors is None:
            num_predecessors = list(len(list(self.DAG.predecessors(t))) for t in self.DAG)
            self.max_task_predecessors = max(num_predecessors)
            self.avg_task_predecessors = np.mean(num_predecessors)
        elif self.max_task_predecessors is None:
            num_predecessors = list(len(list(self.DAG.predecessors(t))) for t in self.DAG)
            self.max_task_predecessors = max(num_predecessors)
        elif self.avg_task_predecessors is None:
            num_predecessors = list(len(list(self.DAG.predecessors(t))) for t in self.DAG)
            self.avg_task_predecessors = np.mean(num_predecessors)
        if self.num_edges is None:
            self.num_edges = self.DAG.number_of_edges()
        if self.edge_density is None:
            max_edges = (self.num_tasks * (self.num_tasks - 1)) / 2 
            self.edge_density = self.num_edges / max_edges         
            
    def compute_CCR(self, platform):
        """
        Computes and sets the computation-to-communication ratio (CCR) for the DAG on the 
        target platform.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.           
        """
        
        cpu_costs = list(task.CPU_cost.mu for task in self.DAG)
        gpu_costs = list(task.GPU_cost.mu for task in self.DAG)
        mean_compute = sum(cpu_costs) * platform.n_CPUs + sum(gpu_costs) * platform.n_GPUs
        mean_compute /= platform.n_workers
        
        cc_comm, cg_comm, gc_comm, gg_comm = 0, 0, 0, 0
        for task in self.DAG:
            cc_comm += (sum(t.mu for t in task.comm_costs["CC"].values()))
            cg_comm += (sum(t.mu for t in task.comm_costs["CG"].values()))
            gc_comm += (sum(t.mu for t in task.comm_costs["GC"].values()))
            gg_comm += (sum(t.mu for t in task.comm_costs["GG"].values()))
           
        mean_comm = platform.n_CPUs * (platform.n_CPUs - 1) * cc_comm
        mean_comm += platform.n_CPUs * platform.n_GPUs * (cg_comm + gc_comm)
        mean_comm += platform.n_GPUs * (platform.n_GPUs - 1) * gg_comm
        mean_comm /= (platform.n_workers**2)
        
        self.CCR[platform.name]["expected"] = mean_compute / mean_comm  
        
        if self.all_actual_costs_set:        
            cpu_costs = list(task.CPU_cost.actual for task in self.DAG)
            gpu_costs = list(task.GPU_cost.actual for task in self.DAG)
            mean_compute = sum(cpu_costs) * platform.n_CPUs + sum(gpu_costs) * platform.n_GPUs
            mean_compute /= platform.n_workers
            
            cc_comm, cg_comm, gc_comm, gg_comm = 0, 0, 0, 0
            for task in self.DAG:
                cc_comm += (sum(t.actual for t in task.comm_costs["CC"].values()))
                cg_comm += (sum(t.actual for t in task.comm_costs["CG"].values()))
                gc_comm += (sum(t.actual for t in task.comm_costs["GC"].values()))
                gg_comm += (sum(t.actual for t in task.comm_costs["GG"].values()))
               
            mean_comm = platform.n_CPUs * (platform.n_CPUs - 1) * cc_comm
            mean_comm += platform.n_CPUs * platform.n_GPUs * (cg_comm + gc_comm)
            mean_comm += platform.n_GPUs * (platform.n_GPUs - 1) * gg_comm
            mean_comm /= (platform.n_workers**2)
            
            self.CCR[platform.name]["actual"] = mean_compute / mean_comm              
        
    def print_info(self, platform=None, detailed=False, filepath=None):
        """
        Print basic information about the DAG, either to screen or as txt file.
        
        Parameters
        ------------------------
        platform - None/Node object (see Environment.py module)/list
        Compute more specific information about the DAG when executed on the platform (if Node)
        or multiple platforms (if list of Nodes).
        
        detailed - bool
        If True, print information about individual Tasks.
        
        filepath - string
        Destination for txt file.                           
        """
        
        print("--------------------------------------------------------", file=filepath)
        print("DAG INFO", file=filepath)
        print("--------------------------------------------------------", file=filepath)   
        print("Application: {}".format(self.app), file=filepath)
        print("Number of tasks: {}".format(self.num_tasks), file=filepath)
        self.compute_topological_info()
        print("Maximum number of task predecessors: {}".format(self.max_task_predecessors), file=filepath)
        print("Average number of task predecessors: {}".format(self.avg_task_predecessors), file=filepath)
        print("Number of edges: {}".format(self.num_edges), file=filepath)
        print("Edge density: {}".format(self.edge_density), file=filepath)
        
        if platform is not None: 
            exp_cpu_costs = list(task.CPU_cost.mu for task in self.DAG)
            exp_gpu_costs = list(task.GPU_cost.mu for task in self.DAG)
            exp_acc_ratios = list(task.acceleration_ratio["expected"] for task in self.DAG) 
            exp_cpu_mu, exp_cpu_sigma = np.mean(exp_cpu_costs), np.std(exp_cpu_costs)
            print("Expected mean task CPU cost: {}, standard deviation: {}".format(exp_cpu_mu, exp_cpu_sigma), file=filepath)
            exp_gpu_mu, exp_gpu_sigma = np.mean(exp_gpu_costs), np.std(exp_gpu_costs)
            print("Expected mean task GPU cost: {}, standard deviation: {}".format(exp_gpu_mu, exp_gpu_sigma), file=filepath)            
            exp_acc_mu, exp_acc_sigma = np.mean(exp_acc_ratios), np.std(exp_acc_ratios)
            print("Expected mean task acceleration ratio: {}, standard deviation: {}".format(exp_acc_mu, exp_acc_sigma), file=filepath) 
            
            if self.all_actual_costs_set:
                act_cpu_costs = list(task.CPU_cost.actual for task in self.DAG)
                act_gpu_costs = list(task.GPU_cost.actual for task in self.DAG)
                act_acc_ratios = list(task.acceleration_ratio["actual"] for task in self.DAG)
                act_cpu_mu, act_cpu_sigma = np.mean(act_cpu_costs), np.std(act_cpu_costs)
                print("Actual mean task CPU cost: {}, standard deviation: {}".format(act_cpu_mu, act_cpu_sigma), file=filepath)
                act_gpu_mu, act_gpu_sigma = np.mean(act_gpu_costs), np.std(act_gpu_costs)
                print("Actual mean task GPU cost: {}, standard deviation: {}".format(act_gpu_mu, act_gpu_sigma), file=filepath)            
                act_acc_mu, act_acc_sigma = np.mean(act_acc_ratios), np.std(act_acc_ratios)
                print("Actual mean task acceleration ratio: {}, standard deviation: {}".format(act_acc_mu, act_acc_sigma), file=filepath)            
            
            if isinstance(platform, list):
                for p in platform:
                    exp_task_mu = (p.n_GPUs * exp_gpu_mu + p.n_CPUs * exp_cpu_mu) / p.n_workers
                    print("\nExpected mean task comp cost on {} platform: {}".format(p.name, exp_task_mu), file=filepath)
                    try:
                        exp_ccr = self.CCR[p.name]["expected"]
                    except KeyError:                 
                        self.compute_CCR(p)
                        exp_ccr = self.CCR[p.name]["expected"]
                    print("Expected computation-to-communication ratio on {} platform: {}".format(p.name, exp_ccr), file=filepath)
                    exp_mst = self.minimal_serial_time(platform) 
                    print("Expected minimal serial time on {} platform: {}".format(p.name, exp_mst), file=filepath)
                    
                    if self.all_actual_costs_set:
                        act_task_mu = (p.n_GPUs * act_gpu_mu + p.n_CPUs * act_cpu_mu) / p.n_workers
                        print("\nActual mean task comp cost on {} platform: {}".format(p.name, act_task_mu), file=filepath)
                        try:
                            act_ccr = self.CCR[p.name]["actual"]
                        except KeyError:                 
                            self.compute_CCR(p)
                            act_ccr = self.CCR[p.name]["actual"]
                        print("Actual computation-to-communication ratio on {} platform: {}".format(p.name, act_ccr), file=filepath)
                        act_mst = self.minimal_serial_time(platform, actual=True) 
                        print("Actual minimal serial cost on {} platform: {}".format(p.name, act_mst), file=filepath)                        
                    
            else:
                exp_task_mu = (platform.n_GPUs * exp_gpu_mu + platform.n_CPUs * exp_cpu_mu) / platform.n_workers
                print("Expected mean task comp cost on {} platform: {}".format(platform.name, exp_task_mu), file=filepath)
                try:
                    exp_ccr = self.CCR[platform.name]["expected"]
                except KeyError:     
                    self.compute_CCR(platform)
                    exp_ccr = self.CCR[platform]["expected"]
                print("Expected computation-to-communication ratio: {}".format(exp_ccr), file=filepath)            
                exp_mst = self.minimal_serial_time(platform) 
                print("Expected minimal serial time: {}".format(exp_mst), file=filepath)
                
                if self.all_actual_costs_set:
                    act_task_mu = (platform.n_GPUs * act_gpu_mu + platform.n_CPUs * act_cpu_mu) / platform.n_workers
                    print("Actual mean task comp cost on {} platform: {}".format(platform.name, act_task_mu), file=filepath)    
                    try:
                        act_ccr = self.CCR[platform.name]["actual"]
                    except KeyError:     
                        self.compute_CCR(platform)
                        act_ccr = self.CCR[platform]["actual"]
                    print("Actual computation-to-communication ratio: {}".format(act_ccr), file=filepath)            
                    act_mst = self.minimal_serial_time(platform, actual=True) 
                    print("Actual minimal serial time: {}".format(exp_mst), file=filepath)                
                        
        if detailed:
            print("\nTASK INFO:", file=filepath)
            for task in self.DAG:
                print("\nTask ID: {}".format(task.ID), file=filepath)
                if task.entry:
                    print("Entry task.", file=filepath)
                if task.exit:
                    print("Exit task.", file=filepath)
                if task.type:
                    print("Task type: {}".format(task.type), file=filepath) 
                print("CPU time (mean, variance): {}".format(task.CPU_cost), file=filepath)  
                print("GPU time (mean, variance): {}".format(task.GPU_cost), file=filepath)
                print("Expected acceleration ratio: {}".format(task.acceleration_ratio["expected"]), file=filepath)     
                if task.actual_costs_set:
                    print("Actual CPU time: {}".format(task.CPU_cost.actual), file=filepath)
                    print("Actual GPU time: {}".format(task.GPU_cost.actual), file=filepath)
                    print("Actual acceleration ratio: {}".format(task.acceleration_ratio["actual"]), file=filepath)                     
        print("--------------------------------------------------------", file=filepath)             
        
    def draw_graph(self, filepath="graphs"):
        """
        Draws the DAG and saves the image.
        
        Parameters
        ------------------------        
        filepath - string
        Destination for image. 
        Notes
        ------------------------                           
        1. See https://stackoverflow.com/questions/39657395/how-to-draw-properly-networkx-graphs       
        """       
        
        G = deepcopy(self.DAG)        
        G.graph['graph'] = {'rankdir':'TD'}  
        G.graph['node']={'shape':'circle', 'color':'#348ABD', 'style':'filled', 'fillcolor':'#E5E5E5', 'penwidth':'3.0'}
        G.graph['edges']={'arrowsize':'4.0', 'penwidth':'5.0'}       
        A = to_agraph(G)
        
        # Add identifying colors if task types are known.
        for task in G:
            if task.type == "GEMM":
                n = A.get_node(task)  
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#E24A33'
                n.attr['label'] = 'G'
            elif task.type == "POTRF":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#348ABD'
                n.attr['label'] = 'P'
            elif task.type == "SYRK":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#988ED5'
                n.attr['label'] = 'S'
            elif task.type == "TRSM":
                n = A.get_node(task)    
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#FBC15E'
                n.attr['label'] = 'T'       
        
        A.layout('dot')
        A.draw('{}/{}_{}tasks_DAG.png'.format(filepath, self.app.split(" ")[0], self.num_tasks))          
            
    def set_actual_costs(self, src=None, realize=False, QOE=(1.0, 1.0), worst_case=False):
        """
        Sets the "actual" computation and communication costs at runtime. 
        This can be done in one of three ways:
            1. Sample all costs from some saved timing data.
            2. Sample all costs from their specified distributions.
            3. Perturb the expected costs uniformly according to some parameter(s).
        
        Parameters
        ------------------------
        src - None/string 
        If not None, the location of saved timing data.            
        
        comp_certainty - float
        The "confidence" we have in our computation estimates. 
        If comp_certainty = x, then actual Task CPU and GPU comp times are set by sampling
        uniformly at random from the interval [comp_certainty * C, (2 - comp_certainty) * C], 
        where C is any of the costs. 
        Intended to be restricted between 0.0 and 1.0 although this isn't checked.
        
        comm_certainty - float
        The "confidence" we have in our communication estimates. 
        If comm_certainty = x, then actual Task communication times are set by sampling
        uniformly at random from the interval [comm_certainty * C, (2 - comm_certainty) * C], 
        where C is any of the costs. 
        Intended to be restricted between 0.0 and 1.0 although this isn't checked.     
                      
        Notes
        ------------------------
        1. If src is not None, this overrides any input comp_ and comm_certainty values and they are ignored.
        """
        
        if isinstance(src, str):   
            with open(src, 'rb') as file:
                comp_samples, comm_samples = dill.load(file) 
        elif isinstance(src, list):
            comp_samples, comm_samples = src
        
        # TODO: bad practice to use nested defaultdicts for things like this...
        if worst_case:
            if src is None:
                raise ValueError("Called set_actual_costs with worst_case == True but no src specified!")
            worst = defaultdict(lambda : defaultdict(float))
            for k, v in comp_samples["CPU"].items():
                xbar = np.mean(v)
                worst["CPU"][k] = max(v, key=lambda x : abs(x - xbar)) 
            for k, v in comp_samples["GPU"].items():
                xbar = np.mean(v)
                worst["GPU"][k] = max(v, key=lambda x : abs(x - xbar))
            for k, v in comm_samples["CC"].items():
                xbar = np.mean(v)
                worst["CC"][k] = max(v, key=lambda x : abs(x - xbar))
            for k, v in comm_samples["CG"].items():
                xbar = np.mean(v)
                worst["CG"][k] = max(v, key=lambda x : abs(x - xbar))
            for k, v in comm_samples["GC"].items():
                xbar = np.mean(v)
                worst["GC"][k] = max(v, key=lambda x : abs(x - xbar))
            for k, v in comm_samples["GG"].items():
                xbar = np.mean(v)
                worst["GG"][k] = max(v, key=lambda x : abs(x - xbar))
            
        for task in self.DAG:
            if src is not None:
                if worst_case:
                    task.CPU_cost.actual = worst["CPU"][task.type]
                    task.GPU_cost.actual = worst["GPU"][task.type]
                else:
                    task.CPU_cost.set_actual_cost(sample=comp_samples["CPU"][task.type])  
                    task.GPU_cost.set_actual_cost(sample=comp_samples["GPU"][task.type]) 
            else:
                task.CPU_cost.set_actual_cost(realize=realize, alpha=QOE[0])
                task.GPU_cost.set_actual_cost(realize=realize, alpha=QOE[0])  
                
            # Set the acceleration ratio.
            task.acceleration_ratio["actual"] = task.CPU_cost.actual / task.GPU_cost.actual
            
            # Set new communication costs.
            for child in self.DAG.successors(task):
                for comm_type in ["CC", "CG", "GC", "GG"]:
                    if src is not None:  
                        if worst_case:
                            task.comm_costs[comm_type][child.ID].actual = worst[comm_type][child.type]
                        else:
                            task.comm_costs[comm_type][child.ID].set_actual_cost(sample=comm_samples[comm_type][child.type])
                    else:
                        task.comm_costs[comm_type][child.ID].set_actual_cost(realize=realize, alpha=QOE[1])            
            
            if not task.actual_costs_set:
                task.actual_costs_set = True
            if task.is_static:
                task.is_static = False
                
        if not self.all_actual_costs_set:
            self.all_actual_costs_set = True 
    
    def save_costs(self, dest=None):
        """
        Save a copy of the current DAG costs.
        Often useful.
        
        Parameters
        ------------------------
        actual - bool 
        If True, save the actual costs. Otherwise, save the expected costs.
        
        dest - None/string
        Destination to save the costs using dill.
        
        Returns
        ------------------------
        costs - list [comp, comm]
        An object that allows the costs to be recovered - e.g., by load_costs() below.
        Intended usage of computation costs is e.g., comp["CPU"][task.ID] = task actual/expected CPU cost.
        Likewise, e.g., comm["CG"][parent.ID][child.ID] = actual/expected CPU-GPU communication cost
        from parent to child Tasks.        
        """
        
        comp = defaultdict(lambda: defaultdict(RV))
        comm = defaultdict(lambda: defaultdict(lambda: defaultdict(RV)))        
            
        for task in self.DAG:
            comp["C"][task.ID] = task.CPU_cost
            comp["G"][task.ID] = task.GPU_cost
            for child in self.DAG.successors(task):
                comm["CC"][task.ID][child.ID] = task.comm_costs["CC"][child.ID] 
                comm["CG"][task.ID][child.ID] = task.comm_costs["CG"][child.ID]  
                comm["GC"][task.ID][child.ID] = task.comm_costs["GC"][child.ID]  
                comm["GG"][task.ID][child.ID] = task.comm_costs["GG"][child.ID]  
                
        costs = [comp, comm]
        if dest:
            with open(dest, 'wb') as handle:
                dill.dump(costs, handle) 
        return costs       
                
    def load_costs(self, src):
        """
        Set either the actual or expected costs of the DAG to the values given by
        the object located at src.
        
        Parameters
        ------------------------
        actual - bool 
        If True, set the actual costs. Otherwise, set the expected costs.
        
        src - string
        Location to load the costs from.               
        """
        
        if isinstance(src, str):
            with open(src, 'rb') as file:
                comp, comm = dill.load(file) 
        else:
            comp, comm = src
            
        for task in self.DAG:            
            task.CPU_cost = comp["C"][task.ID]
            task.GPU_cost = comp["G"][task.ID] 
            for child in self.DAG.successors(task):
                task.comm_costs["CC"][child.ID] = comm["CC"][task.ID][child.ID] 
                task.comm_costs["CG"][child.ID] = comm["CG"][task.ID][child.ID] 
                task.comm_costs["GC"][child.ID] = comm["GC"][task.ID][child.ID] 
                task.comm_costs["GG"][child.ID] = comm["GG"][task.ID][child.ID]                                                                                         
                
    def all_tasks_scheduled(self):
        """Check if all the tasks in the DAG have been scheduled. Returns True if they have, False if not."""
        return all(task.scheduled for task in self.DAG)
    
    def reset(self):
        """ Resets some attributes to defaults so comp of the DAG object can be simulated again. """
        for task in self.DAG:
            task.reset()    
    
    def get_ready_tasks(self):
        """
        Identify the tasks that are ready to schedule.               
        Returns
        ------------------------                          
        List
        All tasks in the DAG that are ready to be scheduled.                 
        """       
        return list(t for t in filter(lambda t: t.ready_to_schedule(self) == True, self.DAG))   
    
    def minimal_serial_time(self, platform, actual=False, use_clark=False):
        """
        Computes the actual or expected minimum makespan of the DAG on a single Worker of the platform.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.        
        
        actual - bool
        If True, use actual rather than expected costs.
        Returns
        ------------------------                          
        float
        The minimal serial time.      
        
        Notes
        ------------------------                          
        1. Assumes all task CPU and GPU costs are set.        
        """  
        if actual:
            return min(sum(task.CPU_cost.actual for task in self.DAG), sum(task.GPU_cost.actual for task in self.DAG)) 
        if use_clark:
            cpu_exp = sum(task.CPU_cost for task in self.DAG)
            gpu_exp = sum(task.GPU_cost for task in self.DAG)
            return cpu_exp.clark_min(gpu_exp)   
        cpu_exp = sum(task.CPU_cost.mu for task in self.DAG)
        gpu_exp = sum(task.GPU_cost.mu for task in self.DAG)
        return min(cpu_exp, gpu_exp)
                   
    def sort_by_upward_rank(self, platform, avg_type="HEFT", cost_type="expected", return_rank_values=False):
        """
        Sorts all tasks in the DAG by decreasing/non-increasing order of upward rank.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
        
        avg_type - string
        How the task and edge weights should be set in platform.average_comm_cost and task.average_comp_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        return_rank_values - bool
        If True, method also returns the upward rank values for all tasks.        
        Returns
        ------------------------                          
        priority_list - list
        Scheduling list of all Task objects prioritized by upward rank.
        
        If return_rank_values == True:
        task_ranks - dict
        Gives the actual upward ranks of all tasks in the form {task : rank_u}.
        
        Notes
        ------------------------ 
        1. "Upward rank" is also called "bottom-level".        
        """             
        
        # Traverse the DAG starting from the exit task.
        backward_traversal = list(reversed(list(nx.topological_sort(self.DAG))))        
        # Compute the upward rank of all tasks recursively, starting with the exit task.
        task_ranks = {}
        for t in backward_traversal:
            task_ranks[t] = t.average_comp_cost(platform, avg_type=avg_type, cost_type=cost_type) 
            if not t.exit:
                if cost_type == "stochastic":
                    children = list(self.DAG.successors(t))
                    c = children[0]
                    m = platform.average_comm_cost(t, c, cost_type="stochastic")
                    m += task_ranks[c] 
                    for child in children[1:]:
                        m1 = platform.average_comm_cost(t, child, cost_type="stochastic")
                        m1 += task_ranks[child] 
                        m = m.clark_max(m1) 
                    task_ranks[t] += m
                else:
                    task_ranks[t] += max(platform.average_comm_cost(parent=t, child=c, avg_type=avg_type, cost_type=cost_type) 
                                         + task_ranks[c] for c in self.DAG.successors(t))
        
        if cost_type == "stochastic":
            priority_list = list(reversed(sorted(task_ranks, key=lambda t : task_ranks[t].mu))) 
        else:            
            priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))            
            
        # Return the tasks sorted in nonincreasing order of upward rank. 
        if return_rank_values:
            return priority_list, task_ranks
        return priority_list     
    
    def sort_by_downward_rank(self, platform, avg_type="HEFT", cost_type="expected", return_rank_values=False):
        """
        Sorts all tasks in the DAG by increasing/non-decreasing order of downward rank.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.
        
        avg_type - string
        How the task and edge weights should be set in platform.average_comm_cost and task.average_comp_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        return_rank_values - bool
        If True, method also returns the downward rank values for all tasks.
        Returns
        ------------------------                          
        priority_list - list
        Scheduling list of all Task objects prioritized by downward rank.
        
        If return_rank_values == True:
        task_ranks - dict
        Gives the actual downward ranks of all tasks in the form {task : rank_d}.
        
        Notes
        ------------------------ 
        1. "Downward rank" is also called "top-level".        
        """        
        
        # Traverse the DAG starting from the entry task.
        forward_traversal = list(nx.topological_sort(self.DAG))        
        # Compute the downward rank of all tasks recursively.
        task_ranks = {}
        for t in forward_traversal:
            task_ranks[t] = RV() if cost_type == "stochastic" else 0
            if not t.entry:
                if cost_type == "stochastic":
                    parents = list(self.DAG.predecessors(t))
                    p = parents[0]
                    m = p.average_comp_cost(platform, avg_type=avg_type, cost_type=cost_type) 
                    m += platform.average_comm_cost(p, t, cost_type="stochastic")
                    m += task_ranks[p]
                    for parent in parents[1:]:
                        m1 = parent.average_comp_cost(platform, avg_type=avg_type, cost_type=cost_type)
                        m1 += platform.average_comm_cost(parent, t, cost_type="stochastic")
                        m1 += task_ranks[parent] 
                        m = m.clark_max(m1) 
                    task_ranks[t] += m
                else:
                    task_ranks[t] += max(p.average_comp_cost(platform, avg_type=avg_type, cost_type=cost_type) 
                    + platform.average_comm_cost(parent=p, child=t, avg_type=avg_type, cost_type=cost_type) 
                    + task_ranks[p] for p in self.DAG.predecessors(t))
       
        if cost_type == "stochastic":
            priority_list = list(sorted(task_ranks, key=lambda t : task_ranks[t].mu))
        else:            
            priority_list = list(sorted(task_ranks, key=task_ranks.get))      
                        
        # Return the tasks sorted in nonincreasing order of upward rank. 
        if return_rank_values:
            return priority_list, task_ranks
        return priority_list 

    def optimistic_cost_table(self, platform):
        """
        Computes the Optimistic Cost Table, as defined in Arabnejad and Barbosa (2014), for the given platform.
        Used in the PEFT heuristic - see Static_Heuristics.py.
        
        Parameters
        ------------------------
        platform - Node object (see Environment.py module)
        The target platform.        
        Returns
        ------------------------                          
        OCT - Nested defaultdict
        The optimistic cost table in the form {Task 1: {Worker 1 : c1, Worker 2 : c2, ...}, ...}.             
        """          
        
        OCT = defaultdict(lambda: defaultdict(float))  
        # Traverse the DAG starting from the exit task(s).
        backward_traversal = list(reversed(list(nx.topological_sort(self.DAG))))
        for task in backward_traversal:
            if task.exit:
                for p in range(platform.n_workers):
                    OCT[task][p] = 0
                continue
            # Not an exit task...
            for p in range(platform.n_workers):
                child_values = []
                for child in self.DAG.successors(task):
                    # Calculate OCT(child, pw) + w(child, pw) for all processors pw.
                    child_CPU_cost = child.CPU_cost.mu
                    child_GPU_cost = child.GPU_cost.mu
                    proc_values = list(OCT[child][pw] + child_CPU_cost if pw < platform.n_CPUs else OCT[child][pw] + child_GPU_cost for pw in range(platform.n_workers))
                    # Add the (approximate) communication cost to the processor value unless pw == p.
                    for pw in range(platform.n_workers):
                        if pw != p: 
                            proc_values[pw] += platform.average_comm_cost(task, child, cost_type="expected") 
                    # Calculate the minimum value over all processors.
                    child_values.append(min(proc_values))
                # OCT is the maximum n of these processor minimums over all the child tasks.
                OCT[task][p] = max(child_values)  
        return OCT       
    
    def accelerated_OCT(self):
        """
        Simpler version of the optimistic cost table for accelerated environments.
        Notes:
            1. Assumes comm costs from processors of same type are always zero, rather than using approximate_comm_cost as in above.
        """
        # Compute the OCT table.
        OCT = defaultdict(lambda: defaultdict(float))  
        # Traverse the DAG starting from the exit task(s).
        backward_traversal = list(reversed(list(nx.topological_sort(self.DAG))))
        for task in backward_traversal:
            if task.exit:
                for p in ["C", "G"]:
                    OCT[task][p] = 0
                continue
            for p in ["C", "G"]:
                child_values = []
                for child in self.DAG.successors(task):
                    child_CPU_cost = child.CPU_cost.mu
                    child_GPU_cost = child.GPU_cost.mu
                    c1, c2 = OCT[child]["C"] + child_CPU_cost, OCT[child]["G"] + child_GPU_cost
                    if p == "G":
                        c1 += task.comm_costs["{}C".format(p)][child.ID].mu
                    else:
                        c2 += task.comm_costs["{}G".format(p)][child.ID].mu
                    child_values.append(min(c1, c2))
                OCT[task][p] = max(child_values) 
        return OCT                    
                                       
    def optimistic_finish_times(self, stochastic=False):
        """
        Computes the optimistic finish time, as defined in the Heterogeneous Optimistic Finish Time (HOFT) algorithm,
        of all tasks assuming they are scheduled on either CPU or GPU. 
        Used in the HOFT heuristic - see Heuristics.py.                  
        Returns
        ------------------------                          
        OFT - Nested defaultdict
        The optimistic finish time table in the form {Task 1: {Worker 1 : c1, Worker 2 : c2, ...}, ...}.         
        
        Notes
        ------------------------ 
        1. No target platform is necessary as parameter since task.CPU_cost and GPU_cost are assumed to be set for all tasks. 
           Likewise, task.comm_costs is assumed to be set for all tasks. 
        """  
                
        d = defaultdict(int)
        d["CC"], d["GG"] = 0, 0
        d["CG"], d["GC"] = 1, 1     
        
        OFT = defaultdict(lambda: defaultdict(float))            
        forward_traversal = list(nx.topological_sort(self.DAG))
        for task in forward_traversal:
#            print("\nTask: {}".format(task.ID))
            for p in ["C", "G"]:
#                print(p)
                if stochastic:
                    OFT[task][p] = task.CPU_cost if p == "C" else task.GPU_cost 
                else:
                    OFT[task][p] = task.CPU_cost.mu if p == "C" else task.GPU_cost.mu                
                if not task.entry:
                    parent_values = []
                    if stochastic:
                        for parent in self.DAG.predecessors(task):
#                            print("Parent: {}".format(parent.ID))
                            action_values = [OFT[parent][q] + d["{}".format(q + p)] * parent.comm_costs["{}".format(q + p)][task.ID] for q in ["C", "G"]]
#                            print("Action values: {}".format(action_values))
                            # Apply Clark minimization equations to action values.
                            a = action_values[0]
                            for b in action_values[1:]:
                                a = a.clark_min(b, rho=0)                        
                            parent_values.append(a) 
#                        print("Parent values: {}".format(parent_values))
                        # Apply Clark maximization equations to parent values.
                        q = parent_values[0] 
                        for r in parent_values[1:]:
                            q = q.clark_max(r, rho=0)                    
                        OFT[task][p] += q
                    else:      
                        for parent in self.DAG.predecessors(task):
                            action_values = [OFT[parent][q] + d["{}".format(q + p)] * parent.comm_costs["{}".format(q + p)][task.ID].mu for q in ["C", "G"]]
                            parent_values.append(min(action_values))                        
                        OFT[task][p] += max(parent_values)   
        return OFT               
    
    def critical_path(self):
        """
        Computes the critical path, a lower bound on the makespan of the DAG.               
        Returns
        ------------------------                          
        cp - float
        The length of the critical path.        
        
        Notes
        ------------------------ 
        1. No target platform is necessary as input since task.CPU_cost and GPU_cost are assumed to be set for all tasks. 
           Likewise, task.comm_costs is assumed to be set for all tasks. 
        2. There are alternative ways to compute the critical path but unlike some others this approach takes
           communication costs into account.
        TODO: stochastic version.
        """              
        
        OFT = self.optimistic_finish_times()
        cp = max(min(OFT[task][processor] for processor in OFT[task]) for task in OFT if task.exit)                         
        return cp       
    
    def make_static(self):
        """
        Converts the task DAG to a static representation by setting all actual costs to their expected counterparts.
        """
        for task in self.DAG:
            task.make_static()
        if not self.all_actual_costs_set:
            self.all_actual_costs_set = True      
            
    def is_static(self):
        """Returns True if all task expected and actual costs are identical, False otherwise."""
        return all(task.is_static for task in self.DAG)
    
    def normalize_costs(self):
        """
        Normalize all expected costs. Take the GPU times to be the default.
        Use this very carefully!        
        """
        
        # Find the largest single cost.        
        max_cost = 0
        for task in self.DAG:
            max_task_cost = max(task.CPU_cost.mu, task.GPU_cost.mu)
            if not task.exit:
                max_task_cost = max(max_task_cost, max(t.mu for t in task.comm_costs["CC"].values()), max(t.mu for t in task.comm_costs["CG"].values()), 
                                    max(t.mu for t in task.comm_costs["GC"].values()), max(t.mu for t in task.comm_costs["GG"].values()))
            max_cost = max(max_cost, max_task_cost)
        
        # Scale by this.
        for task in self.DAG:
            task.CPU_cost /= max_cost
            task.GPU_cost /= max_cost
            
            for child in task.comm_costs["CC"]:
                task.comm_costs["CC"][child] /= max_cost
            for child in task.comm_costs["CG"]:
                task.comm_costs["CG"][child] /= max_cost
            for child in task.comm_costs["GC"]:
                task.comm_costs["GC"][child] /= max_cost
            for child in task.comm_costs["GG"]:
                task.comm_costs["GG"][child] /= max_cost 
                
class Worker:
    """
    Represents any CPU or GPU processing resource. 
    """
    def __init__(self, GPU=False, ID=None):
        """
        Create the Worker object.
        
        Parameters
        --------------------
        GPU - bool
        True if Worker is a GPU. Assumed to be a CPU unless specified otherwise.
        
        ID - Int
        Assigns an integer ID to the task. Often very useful.   
        """  
        self.GPU = True if GPU else False
        self.CPU = not self.GPU   
        self.ID = ID   
        self.load = []
        self.idle = True 
        
    def data_ready_time(self, task, dag, platform, cost_type="expected"):
        """
        Data-ready time (DRT) of a task on a worker.
        """
        
        if task.entry:
            return RV() if cost_type == "stochastic" else 0     
        
        # if cost_type == "actual":
        #     return max(p.FT.actual + platform.comm_cost(p, task, p.where_scheduled, self.ID, cost_type="actual") for p in dag.DAG.predecessors(task))
        
        elif cost_type == "stochastic":        
            parents = list(dag.DAG.predecessors(task))
            parent = parents[0]
            drt = parent.FT
            drt += platform.comm_cost(parent, task, parent.where_scheduled, self.ID, cost_type="stochastic") 
            # Now do the maximization over the other parents (if any).
            for parent in parents[1:]:
                m1 = parent.FT 
                m1 += platform.comm_cost(parent, task, parent.where_scheduled, self.ID, cost_type="stochastic")
                drt = drt.clark_max(m1) 
            return drt
        
        else:
            drt = 0
            for p in dag.DAG.predecessors(task):
                drt = max(drt, p.FT.mu + platform.comm_cost(p, task, p.where_scheduled, self.ID, cost_type=cost_type)) 
            return drt  
        
    def earliest_start_time(self, task, dag, platform, insertion=True, cost_type="expected"):
        """ 
        Returns the estimated earliest start time for a task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node object to which the Worker belongs.
        Needed for calculating communication costs.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        use_actual_costs - bool
        If True, uses Task actual computation and communication times in the computation.
        This is only really intended to be used in the schedule_task method to calculate
        Task AST and AFT attributes (and therefore the "actual" makespan of the DAG),
        so should be used very carefully otherwise.
        
        Returns
        ------------------------
        float 
        The earliest start time for task on Worker.        
        """  
        
        drt = self.data_ready_time(task, dag, platform, cost_type=cost_type)
        
        # If no tasks scheduled on processor...
        if self.idle: 
            return drt                     
                
        # At least one task already scheduled on processor...
        if cost_type == "stochastic":        
            processing_time = task.CPU_cost if self.CPU else task.GPU_cost  
        elif cost_type == "actual":
            processing_time = task.CPU_cost.actual if self.CPU else task.GPU_cost.actual
        else:
            processing_time = task.CPU_cost.mu if self.CPU else task.GPU_cost.mu                              
            
        # Check if it can be scheduled before any task already in the load.  
        if cost_type == "stochastic":
            prev_finish_time = RV()          
            for t in self.load:
                t_st, t_ft = t[2], t[3]
                if t_st.mu < drt.mu:
                    prev_finish_time = t_ft 
                    continue
                poss_start_time = prev_finish_time.clark_max(drt)
                if poss_start_time.mu + processing_time.mu <= t_st.mu:         
                    return poss_start_time
                prev_finish_time = t_ft                 
            # No gap found.
            ft = self.load[-1][3]
            return drt.clark_max(ft)        
        else:
            prev_finish_time = 0            
            for t in self.load:
                t_st = t[2].mu 
                t_ft = t[3].mu
                if t_st < drt:
                    prev_finish_time = t_ft
                    continue
                poss_start_time = max(prev_finish_time, drt)
                if poss_start_time + processing_time <= t_st: 
                    return poss_start_time
                prev_finish_time = t_ft
            # No valid gap found.
            ft = self.load[-1][3].mu
            return max(ft, drt)
    
    def earliest_finish_time(self, task, dag, platform, insertion=True, cost_type="expected"): 
        """
        Returns the estimated earliest finish time for a task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
        
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node to which the Worker belongs. 
        Needed for calculating communication costs.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        use_actual_costs - bool
        If True, uses Task actual computation and communication times in the computation.
        This is only really intended to be used in the schedule_task method to calculate
        Task AST and AFT attributes (and therefore the "actual" makespan of the DAG),
        so should be used very carefully otherwise.           
        
        Returns
        ------------------------
        float 
        The earliest finish time for task on Worker. 
        """ 
        if cost_type == "stochastic":        
            processing_time = task.CPU_cost if self.CPU else task.GPU_cost  
        elif cost_type == "actual":
            processing_time = task.CPU_cost.actual if self.CPU else task.GPU_cost.actual 
        else:
            processing_time = task.CPU_cost.mu if self.CPU else task.GPU_cost.mu            
        return processing_time + self.earliest_start_time(task, dag, platform, insertion=insertion, cost_type=cost_type)        
        
    def schedule_task(self, task, dag, platform, insertion=True, cost_type="expected"):
        """
        Schedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
                
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Node object
        The Node object to which the Worker belongs. 
        Needed for calculating communication costs, although this is a bit unconventional.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        """       
        
        task.scheduled = True
        task.where_scheduled = self.ID      
                        
        # Set task attributes.
        if cost_type == "stochastic":
            task.ST = self.earliest_start_time(task, dag, platform, insertion=insertion, cost_type="stochastic") 
        else:
            mu = self.earliest_start_time(task, dag, platform, insertion=insertion, cost_type=cost_type) # includes actual case.
            task.ST = RV(mu, 0) # TODO: don't like this.            
        
        exp_processing_time = task.CPU_cost if self.CPU else task.GPU_cost 
        
        if cost_type == "stochastic":
            task.FT = task.ST + exp_processing_time
        elif cost_type == "actual":
            task.FT = task.ST + exp_processing_time.actual
        else:
            task.FT = task.ST + exp_processing_time.mu                     
        
        # Add to load.
        if self.idle:
            self.idle = False 
            self.load.append((task.ID, task.type, task.ST, task.FT))
            return
        
        if not insertion:
            self.load.append((task.ID, task.type, task.ST, task.FT))
        else:             
            idx = -1
            for t in self.load:
                if task.ST.mu < t[2].mu:
                    idx = self.load.index(t)
                    break
            if idx > -1:
                self.load.insert(idx, (task.ID, task.type, task.ST, task.FT))
            else:
                self.load.append((task.ID, task.type, task.ST, task.FT))                            

    def unschedule_task(self, task):
        """
        Unschedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a task.                 
        """
        
        # Remove task from the load.
        for t in self.load:
            if t[0] == task.ID:
                self.load.remove(t)
                break
        # Revert the processor to idle if necessary
        if not len(self.load):
            self.idle = True
        # Reset the task itself.    
        task.reset()                         
        
    def print_schedule(self, filepath=None):
        """
        Print the current tasks scheduled on the Worker, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for schedule txt file.                           
        """
        
        proc_type = "CPU" if self.CPU else "GPU"
        print("\nWORKER {}, TYPE: {}: ".format(self.ID, proc_type), file=filepath)      
        for t in self.load:
            type_info = "Type: {}, ".format(t[1]) if t[1] is not None else ""
            print("Task {}, {}START TIME = {}, FINISH TIME = {}.".format(t[0], type_info, t[2], t[3]), file=filepath)            
 
class Node:
    """          
    A Node is basically just a collection of CPU and GPU Worker objects.
    """
    def __init__(self, CPUs, GPUs, name="generic", communication=True):
        """
        Initialize the Node by giving the number of CPUs and GPUs.
        
        Parameters
        ------------------------
        CPUs - int
        The number of CPUs.
        GPUs - int
        The number of GPUs.
        
        name - string
        An identifying name for the Node. Often useful.
        
        communication - bool
        If False, disregard all communication - all costs are taken to be zero.        
        """
        
        self.name = name  
        self.communication = communication             
        self.n_CPUs, self.n_GPUs = CPUs, GPUs 
        self.n_workers = self.n_CPUs + self.n_GPUs
        self.workers = []
        for i in range(self.n_CPUs):
            self.workers.append(Worker(ID=i))          
        for j in range(self.n_GPUs):
            self.workers.append(Worker(GPU=True, ID=self.n_CPUs + j))                             
    
    def print_info(self, filepath=None):
        """
        Print basic information about the Node, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for txt file.                           
        """
        
        print("----------------------------------------------------------------------------------------------------------------", file=filepath)
        print("NODE INFO", file=filepath)
        print("----------------------------------------------------------------------------------------------------------------", file=filepath)
        print("Name: {}".format(self.name), file=filepath)
        print("{} CPUs, {} GPUs".format(self.n_CPUs, self.n_GPUs), file=filepath)
        print("Communication: {}".format(self.communication), file=filepath) 
        print("----------------------------------------------------------------------------------------------------------------\n", file=filepath)     
    
    def reset(self):
        """ Resets some attributes to defaults so we can simulate the execution of another DAG."""
        for w in self.workers:
            w.load = []
            w.idle = True 
            
    def print_schedule(self, heuristic_name="", filepath=None, stochastic=False):
        """
        TODO: actual no longer set...
        Print the current schedule - all tasks scheduled on each Worker - either to screen or as txt file.
        
        Parameters
        ------------------------
        heuristic_name - string
        Name of the heuristic which produced the current schedule. Often helpful.
        
        filepath - string
        Destination for schedule txt file.                           
        """
        
        print("--------------------------------------------------------", file=filepath)
        print("{} SCHEDULE".format(heuristic_name), file=filepath)
        print("--------------------------------------------------------", file=filepath)
        for w in self.workers:
            w.print_schedule(filepath=filepath) 
        if stochastic:
            worker_fts = list(w.load[-1][3] for w in self.workers if w.load)
            try:
                mkspan = worker_fts[0] 
            except IndexError:
                mkspan = RV()
            for ft in worker_fts[1:]:
                mkspan = mkspan.clark_max(ft)            
        else:
            mkspan = max(w.load[-1][3].mu for w in self.workers if w.load)
        print("\nEXPECTED {} MAKESPAN: {}".format(heuristic_name, mkspan), file=filepath)            
        print("--------------------------------------------------------\n", file=filepath)        
            
    def comm_cost(self, parent, child, source_id, target_id, cost_type="expected"):    
        """
        Compute the communication time from a parent task to a child.
        
        Parameters
        ------------------------
        parent - Task object (see Graph.py module)
        The parent task that is sending its data.
        
        child - Task object (see Graph.py module)
        The child task that is receiving data.
        
        source_id - int
        The ID of the Worker on which parent is scheduled.
        
        target_id - int
        The ID of the Worker on which child may be scheduled.
        
        use_actual_costs - bool
        If True, uses Task actual computation and communication times in the computation.
        This is only really intended to be used in the schedule_task method to calculate
        Task AST and AFT attributes (and therefore the "actual" makespan of the DAG),
        so should be used very carefully otherwise.  
        
        Returns
        ------------------------
        float 
        The communication time between parent and child.        
        """  
        
        if source_id == target_id:
            return RV() if cost_type == "stochastic" else 0 
        if not self.communication:
            return RV() if cost_type == "stochastic" else 0  
        
        source_type = "G" if source_id > self.n_CPUs - 1 else "C"
        target_type = "G" if target_id > self.n_CPUs - 1 else "C"
        
        if cost_type == "actual":
            C = parent.comm_costs["{}".format(source_type + target_type)][child.ID].actual
        elif cost_type == "stochastic":
            C = parent.comm_costs["{}".format(source_type + target_type)][child.ID]              
        else:
            C = parent.comm_costs["{}".format(source_type + target_type)][child.ID].mu            
        return C                    
    
    def average_comm_cost(self, parent, child, avg_type="HEFT", cost_type="expected"): 
        """
        Compute the "average" communication time from parent to child tasks. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
        parent - Task object (see Graph.py module)
        The parent task that is sending its data.
        
        child - Task object (see Graph.py module)
        The child task that is receiving data.
        
        avg_type - string
        How the approximation should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", assume each task is on its slowest processor type and compute corresponding communication cost.
            - "simple worst", always use largest possible communication cost.
            - "best", assume each task is on its fastest processor type and compute corresponding communication cost.
            - "simple best", always use smallest possible communication cost.
            - "HEFT-WM", compute mean over all processors, weighted by task acceleration ratios.
            - "PS", "D", "SFB" - speedup-based avg_types from Shetti, Fahmy and Bretschneider, 2013. 
               Returns zero in all three cases so definitions can be found in approximate_comp_cost
               method in the Task class in Graph.py.
                                         
        Returns
        ------------------------
        float 
        The approximate communication cost between parent and child. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou, 2003. 
        """
        
        if not self.communication:
            return RV() if cost_type == "stochastic" else 0
        
        if cost_type == "actual":
            CC = parent.comm_costs["CC"][child.ID].actual
            CG = parent.comm_costs["CG"][child.ID].actual
            GC = parent.comm_costs["GC"][child.ID].actual
            GG = parent.comm_costs["GG"][child.ID].actual
        elif cost_type == "stochastic":
            CC = parent.comm_costs["CC"][child.ID]
            CG = parent.comm_costs["CG"][child.ID]
            GC = parent.comm_costs["GC"][child.ID]
            GG = parent.comm_costs["GG"][child.ID]
        else:        
            CC = parent.comm_costs["CC"][child.ID].mu
            CG = parent.comm_costs["CG"][child.ID].mu
            GC = parent.comm_costs["GC"][child.ID].mu
            GG = parent.comm_costs["GG"][child.ID].mu
            
        if avg_type == "worst" or avg_type == "best": 
            child_CPU_cost, child_GPU_cost = child.CPU_cost.mu, child.GPU_cost.mu 
            parent_CPU_cost, parent_GPU_cost = parent.CPU_cost.mu, parent.GPU_cost.mu                 
        
        if avg_type == "HEFT" or avg_type == "mean":  
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * CC
            c_bar += self.n_CPUs * self.n_GPUs * CG
            c_bar += self.n_CPUs * self.n_GPUs * GC
            c_bar += self.n_GPUs * (self.n_GPUs - 1) * GG
            c_bar /= (self.n_workers**2)
            return c_bar 
        
        elif avg_type == "worst" or avg_type == "WORST": 
            parent_worst_proc = "C" if parent_CPU_cost > parent_GPU_cost else "G"
            child_worst_proc = "C" if child_CPU_cost > child_GPU_cost else "G"
            if parent_worst_proc == "C" and child_worst_proc == "C":
                if self.n_CPUs == 1:
                    return RV(0, 0) if cost_type == "stochastic" else 0
                return CC
            elif parent_worst_proc == "G" and child_worst_proc == "G":
                if self.n_GPUs == 1:
                    return RV(0, 0) if cost_type == "stochastic" else 0
                return GG
            elif parent_worst_proc == "C" and child_worst_proc == "G":
                return CG
            else:
                return GC
                    
        elif avg_type == "simple worst" or avg_type == "SW": 
            if cost_type == "stochastic":
                m = CC
                for m1 in [CG, GC, GG]:
                    m = m.clark_max(m1)
                return m
            return max(CC, CG, GC, GG)
        
        elif avg_type == "best" or avg_type == "BEST": 
            parent_best_proc = "G" if parent_CPU_cost > parent_GPU_cost else "C"
            child_best_proc = "G" if child_CPU_cost > child_GPU_cost else "C"
            if parent_best_proc == child_best_proc:
                return RV(0, 0) if cost_type == "stochastic" else 0
            elif parent_best_proc == "C" and child_best_proc == "G":
                return CG
            else:
                return GC
        
        elif avg_type == "simple best" or avg_type == "sb": 
            if cost_type == "stochastic":
                m = CC
                for m1 in [CG, GC, GG]:
                    m = m.clark_min(m1)
                return m
            return min(CC, CG, GC, GG)         
                
        elif avg_type == "HEFT-WM" or avg_type == "WM":
            A, B = parent.acceleration_ratio["expected"], child.acceleration_ratio["expected"]            
            c_bar = self.n_CPUs * (self.n_CPUs - 1) * CC 
            c_bar += self.n_CPUs * B * self.n_GPUs * CG
            c_bar += A * self.n_GPUs * self.n_CPUs * GC
            c_bar += A * self.n_GPUs * B * (self.n_GPUs - 1) * GG
            c_bar /= ((self.n_CPUs + A * self.n_GPUs) * (self.n_CPUs + B * self.n_GPUs))
            return c_bar          
        
        elif avg_type == "PS" or avg_type == "ps" or avg_type == "diff" or avg_type == "D" or avg_type == "SFB" or avg_type == "sfb": 
            return RV(0, 0) if cost_type == "stochastic" else 0   
        
        raise ValueError('No avg_type (e.g., "mean" or "median") specified for average_comm_cost!')
    
    # TODO: change MCS in heuristics.py. Maybe a lot faster to keep this though? 
    # def follow_schedule(self, dag, schedule, return_info=False, return_graph=False, hybrid=False, schedule_dest=None):
    #     """ 
    #     Schedule all tasks according to the input schedule.
        
    #     Parameters
    #     ------------------------    
    #     dag - DAG object (see Graph.py module)
    #     Represents the task DAG to be scheduled.
              
    #     platform - Node object (see Environment.py module)
    #     Represents the target platform. 
        
    #     partial - bool
    #     Parameter for dag.makespan. If True, assumes schedule is only for a subset of the tasks in the DAG 
    #     and returns makespan as the lastest finish time of all tasks that have been scheduled so far.
        
    #     schedule - dict
    #     An ordered dict {task : Worker ID} which describes where all tasks are to be scheduled and 
    #     in what order this should be done. 
                         
    #     schedule_dest - None/string
    #     Path to save schedule. 
        
    #     Returns
    #     ------------------------
    #     mkspan - float
    #     The makespan of the schedule.        
    #     """   
        
    #     if return_info:
    #         info = {}
        
    #     cost_type = "actual" if hybrid else "expected"
    #     for task in schedule:
    #         chosen_processor = schedule[task]
    #         self.workers[chosen_processor].schedule_task(task, dag, self, cost_type=cost_type)    
    #         if return_info:
    #             p_type = "CPU" if chosen_processor < self.n_CPUs else "GPU"
    #             info[task.ID] = (chosen_processor, p_type, task.type) 
           
    #     if schedule_dest: 
    #         print("The tasks were scheduled in the following order:", file=schedule_dest)
    #         for t in schedule:
    #             print(t.ID, file=schedule_dest)
    #         print("\n", file=schedule_dest)
    #         self.print_schedule(heuristic_name="CUSTOM", filepath=schedule_dest)    

    #     if return_graph:
    #         G = self.convert_schedule_to_DAG(dag, schedule)           
            
    #     # Compute the makespan.
        
    #     # Reset DAG and platform.
    #     dag.reset()
    #     self.reset()   
        
    #     if return_graph:
    #         return G               
    
    def convert_schedule_to_DAG(self, dag, schedule, tasks_scheduled=True, hybrid=False, more_info=False):
        """
        Convert the current load to a PERT-like DAG with only a single RV cost per task/edge.
        """   
        
        if more_info:
            info = {}
        
        # Follow the schedule.
        if not tasks_scheduled:
            cost_type = "actual" if hybrid else "expected"
            for task in schedule:
                chosen_processor = schedule[task]
                self.workers[chosen_processor].schedule_task(task, dag, self, cost_type=cost_type)    
                if more_info:
                    p_type = "C" if chosen_processor < self.n_CPUs else "G"
                    info[task.ID] = (chosen_processor, p_type, task.type) 
        
        # Create the fixed-cost graph.
        G = nx.DiGraph() 
        mapping = {}        
        for task in schedule:
            if task.ID not in mapping:
                n = task.CPU_cost if schedule[task] < self.n_CPUs else task.GPU_cost
                n.ID = task.ID 
                G.add_node(n)
                mapping[task.ID] = n
            else:
                n = mapping[task.ID]                 
                                
            for child in list(dag.DAG.successors(task)):
                if child not in schedule:
                    continue
                if child.ID in mapping:
                    c = mapping[child.ID]
                else:
                    c = child.CPU_cost if schedule[child] < self.n_CPUs else child.GPU_cost
                    c.ID = child.ID
                    G.add_node(c)
                    mapping[child.ID] = c
                    
                G.add_edge(n, c)    
                if schedule[task] != schedule[child]:
                    source_type = "C" if schedule[task] < self.n_CPUs else "G"
                    target_type = "C" if schedule[child] < self.n_CPUs else "G"
                    w = task.comm_costs["{}".format(source_type + target_type)][child.ID]
                else:
                    w = RV()
                G[n][c]['weight'] = w   
                    
        # Add transitive edges if necessary. 
        for task in schedule:            
            n = mapping[task.ID]
            # ancestors = set(nx.algorithms.ancestors(G, n))
            for t in self.workers[schedule[task]].load:
                if t[0] == task.ID:
                    break
                s = mapping[t[0]]
                # if s not in ancestors:
                G.add_edge(s, n)
                G[s][n]['weight'] = RV() 

        # Reset DAG and platform if necessary.
        if not tasks_scheduled:
            dag.reset()
            self.reset()                     
                    
        if more_info:
            return G, info        
        return G
    
    def valid_schedule(self, dag, schedule):
        """
        TODO.
        True if input schedule is valid, False otherwise.
        """
        return

####################################################################################################    
# Other occasionally useful functions.     
#################################################################################################### 
        
def convert_from_dot(dot_path, app=None):
    """
    Create a DAG object from a graph stored as a dot file.
    
    Parameters
    ------------------------
    dot_path - string
    Where the dot file is located.
    app - None/string
    The application that the graph represents, e.g., "Cholesky".   
    Returns
    ------------------------                          
    dag - DAG object
    Converted version of the graph described by the dot file.      
    
    Notes
    ------------------------                          
    1. This is very slow so isn't recommended for even medium-sized DAGs. The vast majority of the time 
       seems to be taken by read_dot (from Networkx) itself, which surely shouldn't be so slow, so I may
       investigate this further in the future.       
    """  
    
    # Use read_dot from Networkx to load the graph.        
    graph = nx.DiGraph(read_dot(dot_path))
    # Check if it's actually a DAG and make the graph directed if it isn't already.
    if graph.is_directed():
        G = graph
    else:
        G = nx.DiGraph()
        G.name = graph.name
        G.add_nodes_from(graph)    
        done = set() 
        for u, v in graph.edges():
            if (v, u) not in done:
                G.add_edge(u, v)
                done.add((u, v))   
        G.graph = deepcopy(graph.graph)
        G.node = deepcopy(graph.node)        
    # Look for cycles.
    try:
        nx.topological_sort(G)
    except nx.NetworkXUnfeasible:
        raise ValueError('Input graph in convert_from_dot has at least one cycle so is not a DAG!')    
    
    # Get app name from the filename (if not input).
    if not app:
        filename = dot_path.split('/')[-1]    
        app = filename.split('.')[0]    
    
    # Create the DAG object.    
    dag = DAG(app=app)
    done = set()     
    for t in nx.topological_sort(G):
        if t not in done:            
            nd = Task()            
            nd.ID = int(t)
            nd.entry = True
            done.add(t)
        else:
            for n in dag.DAG:
                if n.ID == int(t):
                    nd = n    
                    break
        count = 0
        for s in G.successors(t):
            count += 1
            if s not in done:                
                nd1 = Task()                
                nd1.ID = int(s)
                done.add(s) 
            else:
                for n in dag.DAG:
                    if n.ID == int(s):
                        nd1 = n
                        break
            dag.DAG.add_edge(nd, nd1) 
        if not count:
            nd.exit = True  
    dag.num_tasks = len(dag.DAG)      
         
    return dag

def convert_from_nx_graph(graph, app="Random", single_exit=False):
    """
    Create a DAG object from a graph stored as a dot file.
    
    Parameters
    ------------------------
    graph - Networkx Graph (ideally DiGraph)
    The graph to be converted to a DAG object.
    app - None/string
    The application that the graph represents, e.g., "Cholesky".  
    
    single_exit - bool
    If True, add an artificial single exit task.
    Returns
    ------------------------                          
    dag - DAG object
    Converted version of the graph.           
    """ 
    
    # Make the graph directed if it isn't already.
    if graph.is_directed():
        G = graph
    else:
        G = nx.DiGraph()
        G.name = graph.name
        G.add_nodes_from(graph)    
        done = set()
        for u, v in graph.edges():
            if (v, u) not in done:
                G.add_edge(u, v)
                done.add((u, v))  
        G.graph = deepcopy(graph.graph)     
    # Look for cycles...
    try:
        nx.topological_sort(G)
    except nx.NetworkXUnfeasible:
        raise ValueError('Input graph in convert_from_nx_graph has at least one cycle so is not a DAG!')
    
    # Add single exit node if specified.
    if single_exit:
        exits = set(nd for nd in G if not list(G.successors(nd)))
        num_exits = len(exits)
        if num_exits > 1:
            terminus = len(G)
            G.add_node(terminus)
            for nd in G:
                if nd in exits:
                    G.add_edge(nd, terminus)                       
        
    # Create the DAG object.
    dag = DAG(app=app)
    done = set()  
    for t in nx.topological_sort(G):
        if t not in done:           
            nd = Task()                      
            nd.ID = int(t)
            nd.entry = True
            done.add(t)
        else:
            for n in dag.DAG:
                if n.ID == int(t):
                    nd = n 
                    break        
        count = 0
        for s in G.successors(t):
            count += 1
            if s not in done:
                nd1 = Task()                               
                nd1.ID = int(s)
                done.add(s) 
            else:
                for n in dag.DAG:
                    if n.ID == int(s):
                        nd1 = n
                        break
            dag.DAG.add_edge(nd, nd1) 
        if not count:
            nd.exit = True   
    dag.num_tasks = len(dag.DAG)       
        
    return dag

# =============================================================================
# Static heuristics.
# =============================================================================
    
def HEFT(dag, platform, priority_list=None, avg_type="HEFT", cost_type="expected", return_graph=False, schedule_dest=None):
    """
    Heterogeneous Earliest Finish Time.
    'Performance-effective and low-complexity task scheduling for heterogeneous computing',
    Topcuoglu, Hariri and Wu, 2002.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform.  
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
    
    avg_type - string
    How the tasks and edges should be weighted in dag.sort_by_upward_rank.
    Default is "HEFT" which is mean values over all processors as in the original paper. 
    See platform.average_comm_cost and task.average_execution_cost for other options.
    
    return_schedule - bool
    If True, return the schedule computed by the heuristic.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by the heuristic.
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """   
    
    pi = defaultdict(int)
    
    # List all tasks by upward rank unless alternative is specified.
    if priority_list is None:
        priority_list = dag.sort_by_upward_rank(platform, avg_type=avg_type, cost_type=cost_type) 
        
    # Schedule the tasks.        
    for t in priority_list:  
        finish_times = list([p.earliest_finish_time(t, dag, platform, cost_type=cost_type) for p in platform.workers])
        min_processor = np.argmin(finish_times) 
        platform.workers[min_processor].schedule_task(t, dag, platform, cost_type=cost_type)   
        pi[t] = min_processor        
        
    if return_graph:
        G = platform.convert_schedule_to_DAG(dag, pi)
                
    # If schedule_dest, save the schedule.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="HEFT", filepath=schedule_dest)
                            
    # Reset DAG and platform.
    dag.reset()
    platform.reset() 
    
    if return_graph:
        return pi, G
    return pi 

def HOFT(dag, platform, table=None, priority_list=None, return_graph=False, schedule_dest=None):
    """
    Heterogeneous Optimistic Finish Time (HOFT).
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform. 
    
    table - None/Nested defaultdict
    The optimistic finish time table in the form {Task 1: {Worker 1 : c1, Worker 2 : c2, ...}, ...}.
    Computed by dag.optimistic_finish_table if not input.
    Included as parameter because often useful to avoid computing same OFT many times.
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled.   
    
    return_schedule - bool
    If True, return the schedule computed by the heuristic.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by the heuristic.
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """ 
    
    pi = defaultdict(int) 
    
    # Compute OFT table if necessary.
    OFT = table if table is not None else dag.optimistic_finish_times() 
    
    # Compute the priority list if not input.
    if priority_list is None:
        backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
        task_ranks = {}
        for t in backward_traversal:
            task_ranks[t] = max(list(OFT[t].values())) / min(list(OFT[t].values()))
            try:
                task_ranks[t] += max(task_ranks[s] for s in dag.DAG.successors(t))
            except ValueError:
                pass             
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))       
              
    for task in priority_list:                
        finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
        min_p = np.argmin(finish_times)       
        min_type = "C" if min_p < platform.n_CPUs else "G"
        fastest_type = "C" if task.CPU_cost.mu < task.GPU_cost.mu else "G" 
        if min_type == fastest_type:
            platform.workers[min_p].schedule_task(task, dag, platform, cost_type="expected") 
            pi[task] = min_p
        else:
            fastest_p = np.argmin(finish_times[:platform.n_CPUs]) if min_type == "G" else platform.n_CPUs + np.argmin(finish_times[platform.n_CPUs:]) 
            saving = finish_times[fastest_p] - finish_times[min_p]
            # Estimate the costs we expect to incur by scheduling task on min_p and fastest_p.
            min_costs, fastest_costs = 0, 0
            for s in dag.DAG.successors(task):
                s_p = "C" if OFT[s]["C"] < OFT[s]["G"] else "G" # Expectation of where child scheduled based on OFT.
                exec_time = s.CPU_cost.mu if s_p == "C" else s.GPU_cost.mu
                min_costs = max(exec_time + task.comm_costs["{}".format(min_type + s_p)][s.ID].mu, min_costs)
                fastest_costs = max(exec_time + task.comm_costs["{}".format(fastest_type + s_p)][s.ID].mu, fastest_costs)                
            if saving > (min_costs - fastest_costs):
                platform.workers[min_p].schedule_task(task, dag, platform, cost_type="expected")
                pi[task] = min_p
            else:
                platform.workers[fastest_p].schedule_task(task, dag, platform, cost_type="expected") 
                pi[task] = fastest_p
                
    if return_graph:
        G = platform.convert_schedule_to_DAG(dag, pi)
                       
    # If schedule_dest, print the schedule (i.e., the load of all the processors).
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="HOFT", filepath=schedule_dest) 
                    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
    
    if return_graph:
        return pi, G
    return pi  
    
def PEFT(dag, platform, priority_list=None, schedule_dest=None):
    """
    Predict Earliest Finish Time.
    'List scheduling algorithm for heterogeneous systems by an optimistic cost table',
    Arabnejad and Barbosa, 2014.
    
    Parameters
    ------------------------    
    dag - DAG object (see Graph.py module)
    Represents the task DAG to be scheduled.
          
    platform - Node object (see Environment.py module)
    Represents the target platform. 
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
        
    return_schedule - bool
    If True, return the schedule computed by the heuristic.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by the heuristic.
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {Task : ID of Worker it is scheduled on}.    
    """ 
    
    pi = defaultdict(int) 
    
    OCT = dag.optimistic_cost_table(platform)   
    
    if priority_list is not None:    
        for task in priority_list:
            OEFT = [p.earliest_finish_time(task, dag, platform) + OCT[task][p.ID] for p in platform.workers]
            p = np.argmin(OEFT)
            platform.workers[p].schedule_task(task, dag, platform, cost_type="expected")
            pi[task] = p
    else:            
        task_weights = {t.ID : np.mean(list(OCT[t].values())) for t in dag.DAG}    
        ready_tasks = list(t for t in dag.DAG if t.entry)    
        while len(ready_tasks):          
            task = max(ready_tasks, key = lambda t : task_weights[t.ID]) 
            OEFT = [p.earliest_finish_time(task, dag, platform) + OCT[task][p.ID] for p in platform.workers]
            p = np.argmin(OEFT)  
            platform.workers[p].schedule_task(task, dag, platform, cost_type="expected")
            pi[task] = p                
            ready_tasks.remove(task)
            for c in dag.DAG.successors(task):
                if c.ready_to_schedule(dag):
                    ready_tasks.append(c) 
        
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="PEFT", filepath=schedule_dest)        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()        
    return pi      

def CPOP(dag, platform, avg_type="HEFT", schedule_dest=None):
    """
    Critical-Path-on-a-Processor (CPOP).
    'Performance-effective and low-complexity task scheduling for heterogeneous computing',
    Topcuoglu, Hariri and Wu, 2002.
    """  
    
    pi = defaultdict(int)  
    
    # Compute upward and downward ranks of all tasks to find priorities.
    _, upward_ranks = dag.sort_by_upward_rank(platform, avg_type=avg_type, return_rank_values=True)
    _, downward_ranks = dag.sort_by_downward_rank(platform, avg_type=avg_type, return_rank_values=True)
    task_priorities = {t.ID : upward_ranks[t] + downward_ranks[t] for t in dag.DAG}     
    
    # Identify the tasks on the critical path.
    ready_tasks = list(t for t in dag.DAG if t.entry)  
    cp_tasks = set()
    for t in ready_tasks:
        if any(task_priorities[s.ID] - task_priorities[t.ID] < 1e-6 for s in dag.DAG.successors(t)):
            cp = t
            cp_prio = task_priorities[t.ID] 
            cpu_cost, gpu_cost = t.CPU_cost.mu, t.GPU_cost.mu
            break        
    while not cp.exit:
        cp = np.random.choice(list(s for s in dag.DAG.successors(cp) if abs(task_priorities[s.ID] - cp_prio) < 1e-6))
        cp_tasks.add(cp.ID)
        cpu_cost += cp.CPU_cost.mu
        gpu_cost += cp.GPU_cost.mu
    # Find the fastest worker for the CP tasks.
    cp_worker = platform.workers[0] if cpu_cost < gpu_cost else platform.workers[-1]     
       
    while len(ready_tasks):
        task = max(ready_tasks, key = lambda t : task_priorities[t.ID])
        
        if task.ID in cp_tasks:
            cp_worker.schedule_task(task, dag, platform, cost_type="expected")
            pi[task] = cp_worker.ID
        else:
            min_processor = np.argmin([p.earliest_finish_time(task, dag, platform) for p in platform.workers])            
            platform.workers[min_processor].schedule_task(task, dag, platform, cost_type="expected")
            pi[task] = min_processor
    
        ready_tasks.remove(task)
        for c in dag.DAG.successors(task):
            if c.ready_to_schedule(dag):
                ready_tasks.append(c)       
    
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="CPOP", filepath=schedule_dest)        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()      
    return pi

def HEFT_NC(dag, platform, threshold=0.3, schedule_dest=None):
    """
    HEFT No Cross (HEFT-NC).
    'Optimization of the HEFT algorithm for a CPU-GPU environment,'
    Shetti, Fahmy and Bretschneider (2013).
    """
    
    pi = defaultdict(int)  
    
    # Compute all tasks weights.
    _, task_weights = dag.sort_by_upward_rank(platform, avg_type="SFB", return_rank_values=True)           
        
    ready_tasks = list(t for t in dag.DAG if t.entry)    
    while len(ready_tasks):          
        task = max(ready_tasks, key = lambda t : task_weights[t])          
        min_processor = np.argmin([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
        w = task.CPU_cost.mu if min_processor < platform.n_CPUs else task.GPU_cost.mu
        if w == min(task.CPU_cost.mu, task.GPU_cost.mu):
            platform.workers[min_processor].schedule_task(task, dag, platform, cost_type="expected")
            pi[task] = min_processor
        else:
            eft_m = platform.workers[min_processor].earliest_finish_time(task, dag, platform)
            if platform.workers[min_processor].CPU:
                gpu_weights = []
                for p in platform.workers:
                    if p.CPU:
                        continue
                    eft_p = p.earliest_finish_time(task, dag, platform)
                    gpu_weights.append(abs(eft_m - eft_p) / (eft_m / eft_p))
                w_abs = min(gpu_weights)
                fastest_processor = platform.n_CPUs + np.argmin(gpu_weights)                    
            else:
                cpu_weights = []
                for p in platform.workers:
                    if p.GPU:
                        break
                    eft_p = p.earliest_finish_time(task, dag, platform)
                    cpu_weights.append(abs(eft_m - eft_p) / (eft_m / eft_p))
                w_abs = min(cpu_weights)
                fastest_processor = np.argmin(cpu_weights)  
            if task_weights[task] / w_abs <= threshold: 
                platform.workers[min_processor].schedule_task(task, dag, platform, cost_type="expected")
                pi[task] = min_processor
            else:
                platform.workers[fastest_processor].schedule_task(task, dag, platform, cost_type="expected")
                pi[task] = fastest_processor
        ready_tasks.remove(task)
        for c in dag.DAG.successors(task):
            if c.ready_to_schedule(dag):
                ready_tasks.append(c)
            
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="HEFT-NC", filepath=schedule_dest)    
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()        
    return pi

# =============================================================================
# Stochastic heuristics.
# =============================================================================

def MCS(dag, platform, costs_src=None, production_heuristic="HEFT", production_steps=10, selection_steps=10, threshold=0.1, fullahead=True):
    """ 
    TODO: changes to follow_schedule.
    Monte Carlo Scheduling (MCS).
    'Stochastic DAG scheduling using a Monte Carlo approach,'
    Zheng and Sakellariou (2013).
    """
    
    L = [] # List of candidate schedules.
    if not dag.is_static():
        dag.make_static()
    if production_heuristic == "HEFT":
        sched = HEFT(dag, platform)
    elif production_heuristic == "HOFT":
        OFT = dag.optimistic_finish_times() 
        sched = HOFT(dag, platform, table=OFT)
    M_std = platform.follow_schedule(dag, sched, fullahead=fullahead) 
    L.append(sched)
               
    for i in range(production_steps):
        # Simulate actual costs.
        dag.set_actual_costs(src=costs_src, realize=True)        
        if production_heuristic == "HEFT": 
            sched = HEFT(dag, platform, cost_type="actual") 
        elif production_heuristic == "HOFT":
            sched = HOFT(dag, platform, table=OFT, cost_type="actual") 
        # Set actual costs of DAG to expected values.
        dag.make_static()
        M_sched = platform.follow_schedule(dag, sched, fullahead=fullahead)
        M_std = min(M_std, M_sched)
        if sched in L:
            continue
        if M_sched < M_std * (1 + threshold): 
            L.append(sched)
        
    avg_schedule_mkspans = [0.0] * len(L)        
    for i in range(selection_steps):
        # Perturb DAG.
        dag.set_actual_costs(src=costs_src, realize=True)                 
        for j, sched in enumerate(L):
            mkspan = platform.follow_schedule(dag, sched, fullahead=fullahead) 
            avg_schedule_mkspans[j] += mkspan        
    avg_schedule_mkspans[:] = [m / selection_steps for m in avg_schedule_mkspans]
    
    # Find the schedule that minimizes the average makespan.
    return L[np.argmin(avg_schedule_mkspans)]   

def SSTAR(dag, platform, heuristic=HEFT, return_graph=True, schedule_dest=None):
    """
    Based on Stochastic HEFT (SHEFT).        
    'A stochastic scheduling algorithm for precedence constrained tasks on Grid',
    Tang, Li , Liao, Fang, Wu (2011).
    
    However, this function can take other functions, such as HOFT or PEFT, as arguments instead.
    """
    
    # Save original expected values of all costs.
    original_expectations = dag.save_costs()    
    # Modify expected values of all costs to their SHEFT weights.
    for task in dag.DAG:
        task.CPU_cost.mu = task.CPU_cost.approx_weight()
        task.GPU_cost.mu = task.GPU_cost.approx_weight()     
        task.acceleration_ratio["expected"] = task.CPU_cost.mu / task.GPU_cost.mu
        for child in dag.DAG.successors(task):
            task.comm_costs["CC"][child.ID].mu = task.comm_costs["CC"][child.ID].approx_weight() 
            task.comm_costs["CG"][child.ID].mu = task.comm_costs["CG"][child.ID].approx_weight()
            task.comm_costs["GC"][child.ID].mu = task.comm_costs["GC"][child.ID].approx_weight()
            task.comm_costs["GG"][child.ID].mu = task.comm_costs["GG"][child.ID].approx_weight()   
    # Compute schedule by applying HEFT to perturbed DAG.
    pi, G = heuristic(dag, platform, return_graph=return_graph, schedule_dest=schedule_dest)        
    # Reset DAG to original state.
    dag.load_costs(src=original_expectations)
    # Return schedule.
    return pi, G    
    
def SDLS(dag, platform, threshold=0.9, schedule_dest=None):
    """
    Stochastic Dynamic Level Scheduling (SDLS).
    'Scheduling precedence constrained stochastic tasks on heterogeneous cluster systems,'
    Li, Tang, Veeravalli, Li (2015).
    
    TODO: Not sure of the effect of the SDL going below zero - was this considered by the original authors?
    """
    
    # Compute stochastic bottom levels of all tasks.
    _, sb_level = dag.sort_by_upward_rank(platform, cost_type="stochastic", return_rank_values=True)
    # Compute delta values for all tasks.
    deltas = defaultdict(lambda: defaultdict(RV))
    for task in dag.DAG:
        avg_weight = task.average_comp_cost(platform, cost_type="stochastic")  
        deltas["CPU"][task.ID] = avg_weight - task.CPU_cost
        deltas["GPU"][task.ID] = avg_weight - task.GPU_cost     
    
    # Initialize schedule and SDL table.
    pi = defaultdict(int)
    SDL = defaultdict(lambda: defaultdict(RV)) 
    
    # Begin scheduling.
    ready_tasks = list(t for t in dag.DAG if t.entry)    
    while len(ready_tasks): 
        sd_pair_value = -np.inf
        for task in ready_tasks:
            for p in platform.workers:
                # Get delta.
                delta = deltas["CPU"][task.ID] if p.CPU else deltas["GPU"][task.ID]                
                # Compute earliest start time.
                est = p.earliest_start_time(task, dag, platform, cost_type="stochastic")                                      
                # Compute SDL.
                SDL[task.ID][p.ID] = sb_level[task] - est + delta    
                # Keep track of the largest seen so far.                     
                sd_check = norm.ppf(threshold, SDL[task.ID][p.ID].mu, np.sqrt(SDL[task.ID][p.ID].var))
                if sd_check > sd_pair_value:
                    sd_pair_value = sd_check
                    dom_task = task
                    dom_p = p.ID  
        
        # Schedule the task on the chosen processor. 
        platform.workers[dom_p].schedule_task(dom_task, dag, platform)
        pi[dom_task] = dom_p
        
        ready_tasks.remove(dom_task)
        for c in dag.DAG.successors(dom_task):
            if c.ready_to_schedule(dag):
                ready_tasks.append(c) 
    
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="SDLS", filepath=schedule_dest)
    
    # Reset DAG and platform.         
    dag.reset()
    platform.reset() 
    
    return pi

def makespan(G):
    """
    Compute the makespan of a fixed-cost stochastic DAG, assuming actual costs have been set.
    """
    finish_times = defaultdict(float) 
    forward_traversal = list(nx.topological_sort(G))
    for task in forward_traversal:        
        try:
            m = max(G[p][task]['weight'].actual + finish_times[p.ID] for p in G.predecessors(task))
            finish_times[task.ID] = m + task.actual 
        except ValueError:
            finish_times[task.ID] = task.actual                        
    mkspan = finish_times[forward_traversal[-1].ID] # Assumes single exit task.    
    return mkspan 

def Sculli(G):
    """
    Sculli's method for estimating the makespan of a fixed-cost stochastic DAG.
    'The completion time of PERT networks,'
    Sculli (1983).    
    """
    
    finish_times = defaultdict(float) 
    forward_traversal = list(nx.topological_sort(G))
    for task in forward_traversal:
        parents = list(G.predecessors(task))
        try:
            p = parents[0]
            m = G[p][task]['weight'] + finish_times[p.ID] 
            for p in parents[1:]:
                m1 = G[p][task]['weight'] + finish_times[p.ID]
                m = m.clark_max(m1, rho=0)
            finish_times[task.ID] = m + task 
        except IndexError:
            finish_times[task.ID] = task            
    mkspan = finish_times[forward_traversal[-1].ID] # Assumes single exit task.    
    return mkspan       

def corLCA(G, version=2, return_correlation_tree=False):
    """
    CorLCA heuristic for estimating the makespan of a fixed-cost stochastic DAG.
    'Correlation-aware heuristics for evaluating the distribution of the longest path length of a DAG with random weights,' 
    Canon and Jeannot (2016).     
    Assumes single entry and exit tasks.    
    """    
    
    if version == 2:
        # C is basically just F (i.e., finish time) for the correlation tree.
        C = defaultdict(float)
    
    # Like G, correlation tree is a Networkx DiGraph.
    correlation_tree = nx.DiGraph()
    
    # F represents finish times (called Y in 2016 paper). 
    F = defaultdict(float)  
    
    # Traverse the DAG in topological order.
    forward_traversal = list(nx.topological_sort(G))
    for task in forward_traversal:
        
        # Add task to the correlation tree.
        correlation_tree.add_node(task.ID)     
        
        # Find eta.
        dom_parent = None 
        for parent in G.predecessors(task):
            
            # F(parent, task) = start time of task.
            F[(parent.ID, task.ID)] = G[parent][task]['weight'] + F[parent.ID]  
            
            # If version 2, also need to compute possible start times
            if version == 2:
                C[(parent.ID, task.ID)] = G[parent][task]['weight'] + C[parent.ID] 
                
            # Only one parent.
            if dom_parent is None:
                dom_parent = parent 
                eta = F[(parent.ID, task.ID)]
                
            # At least two parents, so need to use Clark's equations to compute eta.
            else:
                F_ij = F[(parent.ID, task.ID)] # More compact...
                
                # Find the lowest common ancestor of the dominant parent and the current parent.
                lca = nx.algorithms.lowest_common_ancestor(correlation_tree, dom_parent.ID, parent.ID)
                
                # Estimate the relevant correlation.
                if version == 2:
                    rho = C[lca].var / (np.sqrt(C[(dom_parent.ID, task.ID)].var) * np.sqrt(C[(parent.ID, task.ID)].var)) 
                else:
                    rho = F[lca].var / (np.sqrt(eta.var) * np.sqrt(F_ij.var)) # TODO: Can be greater than 1...
                
                # Assuming everything normal so make the current parent dominant if its expected value is greater than eta's.
                if F_ij.mu > eta.mu: 
                    dom_parent = parent
                
                # Compute eta.
                eta = eta.clark_max(F_ij, rho=rho) # Variance may decrease here => rho in version 1 can be greater than 1?  
        
        if dom_parent is None: # Entry task...
            F[task.ID] = RV(task.mu, task.var)
            if version == 2:
                C[task.ID] = RV(task.mu, task.var)          
        else:
            F[task.ID] = task + eta              
            # Add edge in correlation tree from the dominant parent to the current task.
            correlation_tree.add_edge(dom_parent.ID, task.ID)
            if version == 2:
                C[task.ID] = task + G[dom_parent][task]['weight'] + C[dom_parent.ID] 
                
    if return_correlation_tree:
        return F[forward_traversal[-1].ID], correlation_tree, C
    return F[forward_traversal[-1].ID] # Assumes single exit task.

def find_closest_point(points, a):
    """
    Helper function for RobHEFT.
    Given a list of points of the form (mean, standard deviation) and an angle a, returns the point which is closest (after scaling)
    to the line through the origin which makes angle a with the horizontal.    
    """   
    
    # Filter the dominated points.   
    points = list(sorted(points, key = lambda p : p[0])) # Ascending order of expected value.
    dominated = [False] * len(points)           
    for i, pt in enumerate(points):
        if dominated[i]:
            continue
        for j, q in enumerate(points[:i]):   
            if dominated[j]:
                continue
            if q[1] < pt[1]:
                dominated[i] = True 
    points = [pt for i, pt in enumerate(points) if not dominated[i]]    
    
    # Convert angle to radians.
    angle = a * np.pi / 180 
    
    # Find max mean and standard deviation in order to scale them all.
    max_mean = max(points, key = lambda pt : pt[0])[0]
    max_std = max(points, key = lambda pt : pt[1])[1] 
    
    # Convert angle to radians.
    angle = a * np.pi / 180 
    # Line segment runs from (0, 0) to (1, tan(a)).
    line_end_pt = np.tan(angle) 
    # Find minimum point.
    min_pt, min_d = points[0], np.inf
    for pt in points:
        # Rescale to fit in the unit square.
        pt0 = pt[0] / max_mean
        pt1 = pt[1] / max_std           
        # Find distance to line.
        d = abs(line_end_pt * pt0 - pt1) / np.sqrt(1 + line_end_pt**2)        
        if d < min_d:
            min_pt, min_d = pt, d    
    return min_pt     

def RobHEFT(dag, platform, angle=45, n_simulations=None, simulation_src=None, schedule_dest=None):  
    """
    RobHEFT (HEFT with robustness) heuristic.
    'Evaluation and optimization of the robustness of DAG schedules in heterogeneous environments,'
    Canon and Jeannot (2010).    
    
    TODO: Faster to build the fixed-cost schedule DAG incrementally than doing the whole thing every time.
    """        
    
    # TODO: not sure from the description what exactly this should be. Maybe email authors?
    # Compute the mean and standard deviation of the upward rank/bottom-level. (Note uses Sculli's method rather than corLCA.)
    # _, bottom_levels = dag.sort_by_upward_rank(platform, cost_type="stochastic", return_rank_values=True) 
    # # Aggregate the expected value and standard deviation according to the angle.
    # max_mu = max([t.mu for t in bottom_levels.values()])
    # max_std = max([np.sqrt(t.var) for t in bottom_levels.values()])
    # task_ranks = {}
    # for t in dag.DAG:
    #     m = bottom_levels[t].mu / max_mu
    #     s = np.sqrt(bottom_levels[t].mu) / max_std
    #     task_ranks[t] = m * np.cos(a) + s * np.sin(a) 
    # priority_list = list(sorted(task_ranks, key=task_ranks.get)) # Ascending order.  
    
    # Just use HEFT ranking for now...
    priority_list = dag.sort_by_upward_rank(platform) 
    
    # Build the schedule as we go.
    current_schedule = defaultdict(int)    
        
    for task in priority_list:
        worker_mkspans = []
        for p in platform.workers:            
            if n_simulations is None:
                current_schedule[task] = p.ID
                G = platform.convert_schedule_to_DAG(dag, current_schedule, actual=False) # TODO...
                mkspan = corLCA(G)                  
                wp = (mkspan.mu, np.sqrt(mkspan.var), p.ID)                
            else:
                current_schedule[task] = p.ID
                mkspans = []
                for _ in range(n_simulations):
                    dag.set_actual_costs(src=simulation_src, realize=True) # Realize RVs if no source to sample from specified.
                    mkspans.append(platform.follow_schedule(dag, current_schedule, partial=True)) 
                wp = (np.mean(mkspans), np.std(mkspans), p.ID)
                
            worker_mkspans.append(wp)
        
        # Find the best worker.
        min_pt = find_closest_point(worker_mkspans, angle)
        best_p = min_pt[2]
        current_schedule[task] = best_p  
            
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in current_schedule:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="RobHEFT", filepath=schedule_dest) 
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()      
    return current_schedule

# =============================================================================
# Experimental.
# =============================================================================

def SHOFT(dag, platform, table=None, prioritization=1, return_graph=False, schedule_dest=None):
    """Stochastic extension of HOFT."""
    
    pi = defaultdict(int) 
    
    # Compute OFT table if necessary.
    OFT = table if table is not None else dag.optimistic_finish_times(stochastic=True)
    
    if prioritization == 1: # Mean only.
#        print(1)
        backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
        task_ranks = {}
        for t in backward_traversal:
            task_ranks[t] = max(OFT[t]["C"].mu, OFT[t]["G"].mu) / min(OFT[t]["C"].mu, OFT[t]["G"].mu)
#            print("t = {}, OFT[t][C] = {}, OFT[t][G] = {}".format(t.ID, OFT[t]["C"], OFT[t]["G"]))
            try:
                task_ranks[t] += max(task_ranks[s] for s in dag.DAG.successors(t))
            except ValueError:
                pass             
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get))) 
    elif prioritization == 2: # UCB-style.
#        print(2)
        backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
        task_ranks = {}
        for t in backward_traversal:
            c = OFT[t]["C"].approx_weight()
            g = OFT[t]["G"].approx_weight()
            task_ranks[t] = max(c, g) / min(c, g)
            try:
                task_ranks[t] += max(task_ranks[s] for s in dag.DAG.successors(t))
            except ValueError:
                pass             
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
    elif prioritization == 3: # Clark max/min of the OFT values.
#        print(3)
        backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
        task_ranks = {}
        for t in backward_traversal:
            ma = OFT[t]["C"].clark_max(OFT[t]["G"])
            mi = OFT[t]["C"].clark_min(OFT[t]["G"])
#            print("t = {}, OFT[t][C] = {}, OFT[t][G] = {}".format(t.ID, OFT[t]["C"], OFT[t]["G"]))
            task_ranks[t] = ma.mu / mi.mu
            try:
                task_ranks[t] += max(task_ranks[s] for s in dag.DAG.successors(t))
            except ValueError:
                pass             
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
    elif prioritization == 4: # approx weight of max/mins computed with Clark eqs.
#        print(4)
        backward_traversal = list(reversed(list(nx.topological_sort(dag.DAG))))
        task_ranks = {}
        for t in backward_traversal:
            ma = OFT[t]["C"].clark_max(OFT[t]["G"])
            mi = OFT[t]["C"].clark_min(OFT[t]["G"])
            mx = ma.approx_weight()
            mn = mi.approx_weight()
            task_ranks[t] = mx / mn
            try:
                task_ranks[t] += max(task_ranks[s] for s in dag.DAG.successors(t))
            except ValueError:
                pass             
        priority_list = list(reversed(sorted(task_ranks, key=task_ranks.get)))
    
    for task in priority_list:                
        finish_times = list([p.earliest_finish_time(task, dag, platform) for p in platform.workers])
        min_p = np.argmin(finish_times)       
        min_type = "C" if min_p < platform.n_CPUs else "G"
        fastest_type = "C" if task.CPU_cost.mu < task.GPU_cost.mu else "G" 
        if min_type == fastest_type:
            platform.workers[min_p].schedule_task(task, dag, platform, cost_type="expected") 
            pi[task] = min_p
        else:
            fastest_p = np.argmin(finish_times[:platform.n_CPUs]) if min_type == "G" else platform.n_CPUs + np.argmin(finish_times[platform.n_CPUs:]) 
            saving = finish_times[fastest_p] - finish_times[min_p]
            # Estimate the costs we expect to incur by scheduling task on min_p and fastest_p.
            min_costs, fastest_costs = 0, 0
            for s in dag.DAG.successors(task):
                s_p = "C" if OFT[s]["C"].mu < OFT[s]["G"].mu else "G" # Expectation of where child scheduled based on OFT.
                exec_time = s.CPU_cost.mu if s_p == "C" else s.GPU_cost.mu
                min_costs = max(exec_time + task.comm_costs["{}".format(min_type + s_p)][s.ID].mu, min_costs)
                fastest_costs = max(exec_time + task.comm_costs["{}".format(fastest_type + s_p)][s.ID].mu, fastest_costs)                
            if saving > (min_costs - fastest_costs):
                platform.workers[min_p].schedule_task(task, dag, platform, cost_type="expected")
                pi[task] = min_p
            else:
                platform.workers[fastest_p].schedule_task(task, dag, platform, cost_type="expected") 
                pi[task] = fastest_p
                
    if return_graph:
        G = platform.convert_schedule_to_DAG(dag, pi)
                       
    # If schedule_dest, print the schedule (i.e., the load of all the processors).
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_schedule(heuristic_name="SHOFT", filepath=schedule_dest) 
                    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
    
    if return_graph:
        return pi, G
    return pi  
    


