import networkx as nx
import numpy as np
import pandas as pd
import sys
import operator
from multiprocessing import Pool
from termcolor import colored

import PTVA_LI_EMPIRICAL.source_estimation as se



def ptva_li_empirical(graph, obs_time, distribution) :
    nb_diffusions = int(np.sqrt(len(list(graph.nodes()))))
    ### Gets the sorted observers and the referential observer (closest one)
    sorted_obs = sorted(obs_time.items(), key=operator.itemgetter(1))
    sorted_obs = [x[0] for x in sorted_obs]
    path_lengths = preprocess(sorted_obs, graph, distribution, nb_diffusions)
    ### Run the estimation
    s_est, likelihoods = se.ml_estimate(graph, obs_time, path_lengths)

    ranked = sorted(likelihoods.items(), key=operator.itemgetter(1), reverse=True)

    return (s_est, ranked)



def preprocess(observers, graph, distr, nb_diffusions):
    graph_copy = graph.copy()
    path_lengths = pd.DataFrame()
    for diff in range(len(observers) + 80):
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
