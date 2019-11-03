import networkx as nx
import numpy as np
import sys

import PTVA_LI_EMPIRICAL_COVARIANCE_AND_MEAN.source_estimation as se
import operator

from multiprocessing import Pool
from termcolor import colored

### Compute a batch in parallel
def ptva_li_empirical_cov_mean(graph, obs_time, distribution) :
    mu = distribution.mean()
    sigma = distribution.std()
    obs = np.array(list(obs_time.keys()))


    path_lengths = {}
    paths = {}

    graph = preprocess(obs, graph, distribution)

    for o in obs:
        path_lengths[o], paths[o] = nx.single_source_dijkstra(graph, o)

    ### Run the estimation
    s_est, likelihoods, d_mu, cov = se.ml_estimate(graph, obs_time, sigma, mu, paths,
        path_lengths)

    ranked = sorted(likelihoods.items(), key=operator.itemgetter(1), reverse=True)

    return (s_est, ranked)

def preprocess(observer, graph, distr):
    for o in observer:
        ### Initialization of the edge delay
        edges = graph.edges()
        for (u, v) in edges:
            graph[u][v]['weight'] = graph[u][v]['weight'] + abs(distr.rvs())

    for (u, v) in edges:
        graph[u][v]['weight'] = graph[u][v]['weight'] / len(observer)
    return graph

# -----------------------------------------------------------------------------
