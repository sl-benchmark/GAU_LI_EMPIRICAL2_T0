import networkx as nx
import numpy as np
import pandas as pd
import sys
import operator
from multiprocessing import Pool
from termcolor import colored

import GAU_LI_EMPIRICAL.source_estimation as se

DIFFUSION = 150

'''
Compute the source estimation.
PARAMETERS:
    - graph: netwrokx graph without any weight
    - obs_time: disctionnary containing observer node --> infected time for the given node
    - distribution: scipy.stats object representing the edge delay distribution
OUTPUT:
    - s_est: the unique source estimation
    - scores: dictionnary containing node -> score of that node
'''
def gau_li_empirical(graph, obs_time, distribution) :
    nb_diffusions = int(np.sqrt(len(list(graph.nodes()))))
    ### Gets the sorted observers and the referential observer (closest one)
    sorted_obs = sorted(obs_time.items(), key=operator.itemgetter(1))
    sorted_obs = [x[0] for x in sorted_obs]
    path_lengths = preprocess(sorted_obs, graph, distribution, nb_diffusions)
    ### Run the estimation
    s_est, scores = se.ml_estimate(graph, obs_time, path_lengths)

    return (s_est, scores)

'''
Make a certain number of diffusion in order approximate the path length between any node to a observer.
PARAMETERS:
    - observers: list of observers
    - graph: unweighted netwrokx graph
    - distr: scipy.stats object representing the edge delay distribution
    - nb_diffusions: number of  diffusions that have to be made
OUTPUT:
    - path_lengths: Pandas dataframe representing the path length between a node (present in the rows
    of the dataframe) and a observer node (present in the column of the dataframe) for every diffusion.
'''
def preprocess(observers, graph, distr, nb_diffusions):
    graph_copy = graph.copy()
    path_lengths = pd.DataFrame()
    for diff in range(DIFFUSION):
        path_lengths_temp = pd.DataFrame()
        ### edge delay
        edges = graph_copy.edges()
        for (u, v) in edges:
            graph_copy[u][v]['weight'] = abs(distr.rvs())
        for o in observers:
            ### Computation of the shortest paths from every observer to all other nodes
            path_lengths_temp[str(o)] = pd.Series(nx.single_source_dijkstra_path_length(graph_copy, o))
        path_lengths = path_lengths.append(path_lengths_temp, sort = False)
        path_lengths.reset_index(inplace = True)
        path_lengths = path_lengths.rename({'index': 'node'}, axis = 1).set_index('node')
    return path_lengths

# -----------------------------------------------------------------------------
