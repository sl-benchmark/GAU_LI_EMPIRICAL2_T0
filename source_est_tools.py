"""This module contains several utility functions"""

import collections
import itertools
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import operator

# ---------------------------- MU VECTORS and COV MATRIX

'''
Compute the mean shortest path of every diffusion
PARAMETERS:
    path_lengths:(pandas.DataFrame) containing all shortest path from every diffusion
RETURN: dictionnary of dictionnary: {obs: {node: mean length}}
'''
def compute_mean_shortest_path(path_lengths):
    path_lengths = path_lengths.reset_index()
    path_lengths = path_lengths.rename({'index': 'node'}, axis = 1).set_index('node')
    return path_lengths.groupby(['node']).mean().to_dict()


def mu_vector_s(path_lengths, s, obs, ref_obs):
    """compute the mu vector for a candidate s

       obs is the ordered list of observers
    """
    v = list()
    for l in range(1, len(obs)):
        #the shortest path are contained in the bfs tree or at least have the
        #same length by definition of bfs tree
        v.append(path_lengths[str(obs[l])][s] - path_lengths[str(ref_obs)][s])
    #Transform the list in a column array (needed for source estimation)
    mu_s = np.zeros((len(obs)-1, 1))
    v = v.sort()
    print('v', v)
    v = v[:11]
    mu_s[:, 0] = v
    return mu_s


def cov_matrix(path_lengths, sorted_obs, s, ref_obs):
    ref_time = path_lengths[str(ref_obs)].loc[s]
    ref_time = np.tile(ref_time, (len(sorted_obs)-1, 1))

    return np.cov(path_lengths.transpose().drop([str(ref_obs)]).reset_index()[s].to_numpy() - ref_time, ddof = 0)



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
