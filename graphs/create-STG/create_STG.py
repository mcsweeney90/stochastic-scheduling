#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create task DAG objects for STG graphs.
"""

import os, pathlib, dill, re
import networkx as nx
from timeit import default_timer as timer
import sys
sys.path.append('../../')   
from src import TDAG

####################################################################################################

# Variables etc used throughout.
size = 100
src_path = 'original/{}'.format(size)
dest = '../STG/{}'.format(size)
pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

####################################################################################################

start = timer()
# Read stg files.
s = 0
for orig in os.listdir(src_path):    
    if orig.endswith('.stg'):  
        s += 1
        print("\n{}".format(orig))
        G = nx.DiGraph()       
        with open("{}/{}".format(src_path, orig)) as f:
            next(f) # Skip first line.            
            for row in f:
                if row[0] == "#":                   
                    break
                # Remove all whitespace - there is probably a nicer way to do this...
                info = " ".join(re.split("\s+", row, flags=re.UNICODE)).strip().split() 
                # Create task. 
                nd = int(info[0])
                if info[2] == '0':
                    G.add_node(nd)
                    continue
                # Add connections to predecessors.
                predecessors = list(n for n in G if str(n) in info[3:])
                for p in predecessors:
                    G.add_edge(p, nd)
        # Convert G to a TDAG object and save. 
        D = TDAG(G)    
                          
        # Save DAG.
        with open('{}/{}.dill'.format(dest, s), 'wb') as handle:
            dill.dump(D, handle)        
        
elapsed = timer() - start     
print("Time taken: {} seconds".format(elapsed))  