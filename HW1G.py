# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:03:17 2016
Group 4
Network Analytics HW1
"""
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


G = nx.read_weighted_edgelist('HW1_problem1.txt', nodetype = str, create_using=nx.DiGraph())

adj = nx.adjacency_matrix(G, nodelist=['a','b','c','d','e'])
print(adj.todense())

inc = nx.incidence_matrix(G, nodelist=['a','b','c','d','e'], oriented=True)
print(inc.todense())

#Print the shortest-path matrix between all pairs
distance_matrix = nx.floyd_warshall_numpy(G, nodelist=['a','b','c','d','e'], weight='weight')
print(distance_matrix)

#Find the longest shortest path in the graph
def findDiam(dm):
    diam = 0
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            if dm[i,j] != np.inf and dm[i,j] >= diam:
                diam = dm[i,j]
    
    return diam 

d = findDiam(distance_matrix)    
print('The diameter of the graph is ' + str(d))
  
#Plot the degree distribution of the graph
degree_values = sorted(nx.degree(G).values())
plt.hist(degree_values)
plt.title("Degree Distribution plot")
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.savefig("degree_histogram.png")
plt.close()

#Check weak and strong connectivity
print(nx.is_strongly_connected(G))
print(nx.is_weakly_connected(G))


#Draw the graph
#We removed the top half of the matrix in the txt file to keep 34 rows
G2 = np.loadtxt('HW1_problem2.txt')
G2 = np.matrix(G2)
G2 = nx.from_numpy_matrix(G2)


pos = nx.spring_layout(G2)
plt.figure(figsize=(15,10))

nx.draw(G2,pos)
edge_weight=dict([((u,v),int(d['weight'])) for u,v,d in G2.edges(data=True)])

nx.draw_networkx_edge_labels(G2,pos,edge_labels=edge_weight)
nx.draw_networkx_nodes(G2,pos)
nx.draw_networkx_edges(G2,pos)
nx.draw_networkx_labels(G2,pos)
    
plt.savefig("graph_plot.png")    
plt.close()
