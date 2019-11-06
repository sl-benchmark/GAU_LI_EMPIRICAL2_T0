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
    obs = np.array(list(obs_time.keys()))

    path_lengths = preprocess(obs, graph, distribution, nb_diffusions)

    ### Run the estimation
    s_est, likelihoods, d_mu, cov = se.ml_estimate(graph, obs_time, path_lengths)

    ranked = sorted(likelihoods.items(), key=operator.itemgetter(1), reverse=True)

    return (s_est, ranked)

def preprocess(observer, graph, distr, nb_diffusions):
    graph_copy = graph.copy()
    path_lengths = pd.DataFrame()
    for diff in range(nb_diffusions):
        path_lengths_temp = pd.DataFrame()
        ### edge delay
        edges = graph_copy.edges()
        for (u, v) in edges:
            graph_copy[u][v]['weight'] = abs(distr.rvs())
        for o in observer:
            ### Computation of the shortest paths from every observer to all other nodes
            path_lengths_temp[str(o)] = pd.Series(nx.single_source_dijkstra_path_length(graph_copy, o))
        path_lengths = path_lengths.append(path_lengths_temp)
    return path_lengths

# -----------------------------------------------------------------------------
