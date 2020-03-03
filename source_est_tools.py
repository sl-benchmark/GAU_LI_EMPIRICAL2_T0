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

'''
Compute the mean vector for a candidate s.
PARAMETERS:
    - path_lengths: Pandas dataframe representing the path length of every diffusion
    - s: the condidate source
    - obs_list: observer list where the first observer in the list represents the reference observer
    - ref_obs: the reference observer
OUTPUT:
    - mu_s: array representing mean vector
    - obs_list: * If len(obs_list)-1 <= K_0: represents the observer list without the reference observer
                * Else: represents the K_0 closest observers to the candidate source
                        without the reference observer
'''
K_0 = 100
def mu_vector_s(path_lengths, s, obs_list, ref_obs):
    v = list()
    for l in range(1, len(obs_list)):
        #the shortest path are contained in the bfs tree or at least have the
        #same length by definition of bfs tree
        v.append(path_lengths[str(obs_list[l])][s] - path_lengths[str(ref_obs)][s])
    print('len obs', len(obs_list))
    #Transform the list in a column array (needed for source estimation)
    if len(obs_list)-1 <= K_0:
        mu_s = np.zeros((len(obs_list)-1, 1))
        obs_list = obs_list[1:]
    else:
        mu_s = np.zeros((K_0, 1))
        indices = np.array(sorted(range(len(v)), key = lambda sub: v[sub])[:K_0])
        obs_list = np.array(obs_list)[indices+1]
        v = sorted(v)
        v = v[:K_0]
    mu_s[:, 0] = v
    print('mu_s', mu_s)
    print('obs_list', obs_list)
    return mu_s, obs_list

'''
Compute the covariance matrix.
PARAMETERS:
    - path_lengths: Pandas dataframe representing the path length of every diffusion
    - selected_obs: observer list without containing the reference observer
    - s: the candidate source
    - ref_obs: the reference observer
OUTPUT:
    - 2D array representing covariance matrix
'''
def cov_matrix(path_lengths, selected_obs, s, ref_obs):
    ref_time = path_lengths[str(ref_obs)].loc[s]
    ref_time = np.tile(ref_time, (len(selected_obs), 1))
    #return np.cov(path_lengths.transpose().drop([str(ref_obs)]).reset_index()[s].to_numpy() - ref_time, ddof = 0)
    obs_col = [str(s_obs) for s_obs in selected_obs]
    return np.cov(path_lengths[obs_col].transpose().reset_index()[s].to_numpy() - ref_time, ddof = 0)


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
