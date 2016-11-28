# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:57:57 2016

"""
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import numpy as np
import haversine as hv

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