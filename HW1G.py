# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:03:17 2016
"""
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


G = nx.read_weighted_edgelist("HW1_problem1.txt", nodetype = str, create_using=nx.DiGraph())

adj = nx.adjacency_matrix(G, nodelist=['a','b','c','d','e'])
print(adj.todense())

inc = nx.incidence_matrix(G,nodelist=['a','b','c','d','e'],oriented=True)
print(inc.todense())

distance_matrix = nx.floyd_warshall_numpy(G, nodelist=['a','b','c','d','e'], weight='weight')
print(distance_matrix)

#Finding diameter
def findDiam(G):
    diam = 0
    dm = nx.floyd_warshall_numpy(G, nodelist=['a','b','c','d','e'], weight='weight')
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            if dm[i,j] != np.inf and dm[i,j] >= diam:
                diam = dm[i,j]
    
    return diam 

d = findDiam(G)    
print('Diameter of Graph is ' + str(d))
    
#write_dot(G,'hello.dot')
#Generate values for histogram
deg=G.degree()
values = [] #in same order as traversing keys
keys = [] #also needed to preserve order
for key in deg.keys():
  keys.append(key)
  values.append(deg[key])
  
#Plot histogram
plt.hist(values)
plt.title("Degree Distribution plot")
plt.ylabel("Degree")
plt.xlabel("Frequency")
plt.savefig("degree_histogram.png")
plt.close()
#Check weak and strong connectivity
print(nx.is_strongly_connected(G))
print(nx.is_weakly_connected(G))

#Q2
G2 = np.loadtxt('HW1_problem2.txt')
G2 = np.asmatrix(G2)
G2 = nx.from_numpy_matrix(G2)


pos = nx.shell_layout(G2)
plt.figure(figsize=(15,10))

nx.draw(G2,pos)
edge_weight=dict([((u,v,),int(d['weight'])) for u,v,d in G2.edges(data=True)])

nx.draw_networkx_edge_labels(G2,pos,edge_labels=edge_weight)
nx.draw_networkx_nodes(G2,pos)
nx.draw_networkx_edges(G2,pos)
nx.draw_networkx_labels(G2,pos)
    
plt.savefig("graph_plot.png")    
plt.close()
