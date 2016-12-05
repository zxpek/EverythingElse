"""
Created on Thu Nov 28, 2016

Group 4

Network Analytics HW2
"""
from geopy.distance import vincenty
import pandas as pd
import numpy as np
import networkx as nx
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from gurobipy import *
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import haversine as hv


"""
Question 2
"""
# Load file into a numpy matrix
G2 = np.loadtxt('HW2_problem 2.txt')

# Create the graph using the adjacency matrix (skip the top 34 rows)
G2 = nx.from_numpy_matrix(G2[34:68, 0:34])

#Calculate 4 centrality measures
betw = nx.betweenness_centrality(G2, normalized = True)
print(sorted(betw, key = betw.get, reverse = True)[:2])

flow = nx.current_flow_closeness_centrality(G2)
print(sorted(flow, key = flow.get, reverse = True)[:2])

eigenvector = nx.eigenvector_centrality(G2)
print(sorted(eigenvector, key = eigenvector.get, reverse = True)[:2])

closeness = nx.closeness_centrality(G2, normalized = True)
print(sorted(closeness, key = closeness.get, reverse = True)[:2])


"""
Question 3.a.
"""
#Mercator projection
#llcrnrlat: lower left corner latitude (degrees)
#llcrnrlon: lower left corner longitude (degrees)
#urcrnrlat: upper right corner latitude (degrees)
#urcrnrlon: upper right corner longitude (degrees)
#high resolution
#coastline or lake with an area smaller than 0.1km^2 will not be plotted
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
plt.title('Mercator projection')
plt.show()

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

#Plot latitude and longitude scatter plot
x,y = djibouti(lats, lons)
#djibouti.plot(x, y, 'bo', markersize = 3)
plt.plot(x, y, 'bo', markersize=3)
plt.title('Scatter plot')
plt.show()


'''
Question 3c
Traveling Salesman Problem
Code adapted from http://examples.gurobi.com/traveling-salesman-problem/
'''

# Find distance between two points
def distance (df, origin, destination):
	return df.iloc[origin,destination]

# Eliminate subtours that are not visiting all the nodes
def subtour_eliminate(model, where):
	if where == GRB.callback.MIPSOL:
		selected = []
		# list of edges included in solution
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

# Find the shortest subtour
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

# Add constraint to prevent loops and make sure to visit each city exactly once (degree-2 constraint)
for origin in range(n):
  m.addConstr(quicksum(vars[origin,destination] for destination in range(n)) == 2)
  vars[origin,origin].ub = 0

m.update()

# Find optimal tour
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


'''
Question 3b
Use the or-tools traveling salesman routine
'''
f = np.loadtxt('HW2_tsp.txt',skiprows=10)

fLength = f.shape[0]

#Haversine Distances
dist_mat = np.zeros((fLength,fLength))
for i in range(f.shape[0]):
    for j in range(f.shape[0]):
        point1 = (f[i,1]/1000,f[i,2]/1000)
        point2 = (f[j,1]/1000,f[j,2]/1000)
        dist_mat[i,j]=hv.haversine(point1,point2)
        
# Distance callback
class CreateDistanceCallback(object):
  """Create callback to calculate distances between points."""
  def __init__(self):
    """Array of distances between points."""

    self.matrix = dist_mat # Salt Lake City

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]

def main(depot):

  # Cities
  node_list = list(range(len(dist_mat)))

  tsp_size = len(node_list)

  # Create routing model
  if tsp_size > 0:
    # TSP of size tsp_size
    # Second argument = 1 to build a single tour (it's a TSP).
    # Nodes are indexed from 0 to tsp_size - 1. By default the start of
    # the route is node 0.
    routing = pywrapcp.RoutingModel(tsp_size, 1, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Create the distance callback, which takes two arguments (the from and to node indices)
    # and returns the distance between these nodes.

    dist_between_nodes = CreateDistanceCallback()
    dist_callback = dist_between_nodes.Distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    # Solve, returns a solution if any.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
      # Solution cost.
      
      # Inspect solution.
      # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
      route_number = 0
      index = routing.Start(route_number) # Index of the variable for the starting node.
      route = ''
      while not routing.IsEnd(index):
        # Convert variable indices to node indices in the displayed route.
        route += str(node_list[routing.IndexToNode(index)]) + ' -> '
        index = assignment.Value(routing.NextVar(index))
      route += str(node_list[routing.IndexToNode(index)])
      
      return assignment.ObjectiveValue(), route
    else:
      print ('No solution found.')
  else:
    print ('Specify an instance greater than 0.')
    
def solveTSP():    
    tsp_size = 38
       
    dist, route, start = 99999999999, '', 0         
    for i in range(tsp_size):
        a,b = main(i)
        if a < dist:
            dist, route, start = a, b, i
    print ("Total distance: " + str(dist) + " miles\n")
    print ("Route:\n\n" + route)
    return dist, route, start
    
solveTSP()