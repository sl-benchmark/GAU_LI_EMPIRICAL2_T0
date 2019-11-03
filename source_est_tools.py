"""This module contains several utility functions"""

import collections
import itertools
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import operator

# ---------------------------- MU VECTORS and COV MATRIX


def mu_vector_s(paths, s, obs):
    """compute the mu vector for a candidate s

       obs is the ordered list of observers
    """
    v = list()
    for l in range(1, len(obs)):
        #the shortest path are contained in the bfs tree or at least have the
        #same length by definition of bfs tree
        v.append(len(paths[obs[l]][s]) - len(paths[obs[0]][s]))
    #Transform the list in a column array (needed for source estimation)
    mu_s = np.zeros((len(obs)-1, 1))
    mu_s[:, 0] = v
    return mu_s

def verif_existant_path(edges, path):
    """Verifies if the existing path exists in the list of edges.

    edges : list of edges of the current graph
    path : list of 2tuples representing the path

    """
    path_edges = zip(path[:-1], path[1:])
    return all(any(p1==p2 for p1 in edges) for p2 in path_edges)


def mu_vector_s(path_lengths, s, obs, obs_ref):
    mu_vector = []
    for o in obs:
        mu_vector.append(path_lengths[o][s])
    return mu_vector

def cov_mat(graph, path_lengths, sorted_obs):
    cov_matrix = []
    for o1 in sorted_obs:
        temp_list = []
        for o2 in sorted_obs:
            temp_list.append(path_lengths[o1][o2])
        cov_matrix.append(temp_list)
    return np.cov(cov_matrix)





# ---------------------------- Filtering diffusion data

def filter_diffusion_data(infected, obs, max_obs=np.inf):
    """Takes as input two dictionaries containing node-->infection time and filter
     only the items that correspond to observer nodes, up to the max number of observers

     INPUT :
        infected (dict) infection times for every node
        obs (list) list of observers
        max_obs (int) max number of observers to be picked
    """

    ### Filter only observer nodes
    obs_time = dict((k,v) for k,v in infected.items() if k in obs)

    ### If maximum number does not include every observer, we pick the max_obs closest ones
    if max_obs < len(obs_time):

        ### Sorting according to infection times & conversion to a dict of 2tuples
        node_time = sorted(obs_time.items(), key=operator.itemgetter(1), reverse=False)
        new_obs_time = {}

        ### Add max_obs closest ones
        (n, t) = node_time.pop()
        while len(new_obs_time) < max_obs:
            new_obs_time[n] = t
            (n, t) = node_time.pop()

        return new_obs_time

    else:
        return obs_time

# ---------------------------- Equivalence classes

def classes(path_length, sorted_obs):
    """Computes the equivalenc classes among all the graph nodes with
    respect to their distance to all observers in the graph

    INPUT:
        path_length (dict of dict) lengths of shortest paths from each node to all nodes
        sorted_obs (list) list of observers sorted by infection time

    """
    ### Gets first infected observer and initializes the class dict
    min_obs = sorted_obs[0]
    vector_to_n = collections.defaultdict(list) # creates dictionnary that will create an empty list when a non existent key is accessed

    ### Loops over all nodes reachables from the first infected node
    for neighbor in path_length[min_obs].keys():
        ## In short computes key=distance to all observers and value=node
        tuple_index = tuple(int((10**8)*(path_length[observer][neighbor] - path_length[min_obs][neighbor])) for observer in sorted_obs[1:])
        vector_to_n[tuple_index].append(neighbor)

    classes = vector_to_n.values()
    return classes
