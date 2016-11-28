'''
Akos Furton
Assignment 2 - Individual Part
23 November 2016
'''

## PACKAGE IMPORTS
from geopy.distance import vincenty
import pandas as pd
import numpy as np
import networkx as nx
from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from gurobipy import *


with open('HW2_tsp.txt') as lat_lon_file:
	data = lat_lon_file.read().splitlines()
	data = data[10:]
	for x in range(len(data)):
		data[x] = data[x].split()
	for x in range(len(data)):
		data[x][1] = float(data[x][1]) / 1000
		data[x][2] = float(data[x][2]) / 1000

lats = []
for x in range (len(data)):
	lats.append(data[x][1])
lons = []
for x in range (len(data)):
	lons.append(data[x][2])


df = pd.DataFrame(data, columns = ['city','lat', 'lon'])

Dist_Matrix = []
for i in range(len(df)):
	Row = []
	for j in range(len(df)):
		Row.append(vincenty(tuple([df.iat[i,1],df.iat[i,2]]),tuple([df.iat[j,1],df.iat[j,2]])).km)
	Dist_Matrix.append(Row)

distances = pd.DataFrame(Dist_Matrix)
distances.to_csv('Distance_Matrix.csv')


'''
#Question 2
#Calculate 4 centrality measures
#Betweenness,
#Flow,
#Eigenvalue,
#Other
#Two most important nodes for each criteria - what info captured and appropriateness



G2 = np.loadtxt('HW2_problem 2.txt')
G2 = nx.from_numpy_matrix(G2[34:68, 0:34])

betw = nx.betweenness_centrality(G2, normalized = True)
print(sorted(betw, key = betw.get, reverse = True)[:2])

flow = nx.current_flow_closeness_centrality(G2)
print(sorted(flow, key = flow.get, reverse = True)[:2])

eigenvector = nx.eigenvector_centrality(G2)
print(sorted(eigenvector, key = eigenvector.get, reverse = True)[:2])

closeness = nx.closeness_centrality(G2)
print(sorted(closeness, key = closeness.get, reverse = True)[:2])
'''

'''
#Question 3a
#Plot lat/long scatter plot 
'''


djibouti = Basemap(projection = 'merc',
	llcrnrlon = 41,
	llcrnrlat = 10.5,
	urcrnrlon = 44,
	urcrnrlat = 13,
	resolution = 'h',
	area_thresh = 0.1)

djibouti.drawcoastlines()
djibouti.drawcountries()
djibouti.fillcontinents(color = '#EDC9AF', lake_color = 'aquamarine')
djibouti.drawmapboundary(fill_color = 'aquamarine')


x,y = djibouti(lons, lats)
djibouti.plot(x, y, 'bo', markersize = 4)



'''
Question 3c
Travelling Salesman
Code adapted from http://examples.gurobi.com/traveling-salesman-problem/
'''

# Returns distance between two points
def distance (df, origin, destination):
	return df.iloc[origin,destination]

# Eliminates subtours that are not the full route
def subtour_eliminate(model, where):
	if where == GRB.callback.MIPSOL:
		selected = []
		#list of edges included in solution
		for origin in range(n):
			solution = model.cbGetSolution([model._vars[origin,destination] for destination in range(n)])
			selected += [(origin,destination) for destination in range(n) if solution[destination] > 0.5]
		# find shortest cycle in edge list
		tour = subtour(selected)
		if len(tour) < n:
			print()
			# add a subtour elimination constraint
			expr = 0
			for origin in range(len(tour)):
				for destination in range(origin+1, len(tour)):
					expr += model._vars[tour[origin], tour[destination]]
			model.cbLazy(expr <= len(tour)-1)

# Creates a path from a subset of edges
def subtour(edges):
	visited = [False]*n
	cycles = []
	lengths = []
	selected = [[] for i in range(n)]
	for x,y in edges:
		selected[x].append(y)
	while True:
		current = visited.index(False)
		thiscycle = [current]
		while True:
			visited[current] = True
			neighbors = [node for node in selected[current] if not visited[node]]
			if len(neighbors) == 0:
				break
			current = neighbors[0]
			thiscycle.append(current)
		cycles.append(thiscycle)
		lengths.append(len(thiscycle))
		if sum(lengths) == n:
			break
	return (cycles[lengths.index(min(lengths))])

# Create variables
m = Model()
n = len(distances)
vars = {}
for origin in range(n):
   for destination in range(origin+1):
     vars[origin,destination] = m.addVar(obj=distance(distances, origin, destination), vtype=GRB.BINARY,
                          name='e'+str(origin)+'_'+str(destination))
     vars[destination,origin] = vars[origin,destination]
   m.update()

# Add constraint to prevent loops and make sure loop passes through each city
for origin in range(n):
  m.addConstr(quicksum(vars[origin,destination] for destination in range(n)) == 2)
  vars[origin,origin].ub = 0

m.update()

# Travelling Salesman Optimization
m._vars = vars
m.params.LazyConstraints = 1
m.optimize(subtour_eliminate)

solution = m.getAttr('x', vars)
selected = [(origin,destination) for origin in range(n) for destination in range(n) if solution[origin,destination] > 0.5]


'''
Question 3d
Plot TSP optimal solution
'''

for i in selected:
	xs = []
	ys = []

	xpoint0, ypoint0 = djibouti(lons[i[0]], lats[i[0]])
	xs.append(xpoint0)
	ys.append(ypoint0)

	xpoint1, ypoint1 = djibouti(lons[i[1]], lats[i[1]])
	xs.append(xpoint1)
	ys.append(ypoint1)

	djibouti.plot(xs, ys, color = 'grey', linewidth = 1)

plt.savefig('Djibouti.png')
plt.close()

