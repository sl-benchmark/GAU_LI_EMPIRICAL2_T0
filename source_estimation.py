"""This file contains some functions needed to estimate (via maximum
likelihood) the source of a SI epidemic process (with Gaussian edge delays).

The important function is
    s_est, likelihood = ml_estimate(graph, obs_time, sigma, is_tree, paths,
    path_lengths, max_dist)

where s_est is the list of nodes having maximum a posteriori likelihood and
likelihood is a dictionary containing the a posteriori likelihood of every
node.

"""
import os, sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

import math
import random
import networkx as nx
import numpy as np
import source_est_tools as tl
import operator
import collections

import scipy.stats as st
from scipy.misc import logsumexp

def ml_estimate(graph, obs_time, path_lengths, max_dist=np.inf):

    """Returns estimated source from graph and partial observation of the
    process.

    - graph is a networkx graph
    - obs_time is a dictionary containing the observervations: observer -->
      time

    Output:
    - list of nodes having maximum a posteriori likelihood
    - dictionary: node -> a posteriori likelihood

    """

    ### Gets the referential observer took at random
    obs_list = obs_time.keys()

    ### Gets the nodes of the graph and initializes likelihood
    nodes = np.array(list(graph.nodes))
    loglikelihood = {n: -np.inf for n in nodes}

    # average the path lengths from all the diffusion
    mean_path_lengths = tl.compute_mean_shortest_path(path_lengths)

    # candidate nodes does not contain observers nodes by assumption
    candidate_nodes = np.array(list(set(nodes) - set(obs_list)))

    for s in candidate_nodes:
        # covariance matrix
        cov_d_s = tl.cov_matrix(path_lengths, obs_list, s)
        cov_d_s_inv = np.linalg.inv(cov_d_s)
        
        ### vector -> difference between observation time and mean arrival time for observers
        w_s = list()
        for obs in obs_time.keys():
            w_s.append(obs_time[obs] -  mean_path_lengths[str(obs)][s])

        I = np.ones((len(w_s)))

        ### MLE of initial time t0
        t0_s = ((I.T @ cov_d_s_inv @ w_s) / (I.T @ cov_d_s_inv @ I))

        ### Auxilary variable to make equation simpler to write
        z_s = ((w_s - (t0_s*I)).T) @ cov_d_s_inv @ (w_s - (t0_s*I))

        ### estimator for the source node
        loglikelihood[s] = -(len(obs_list)*np.log(z_s) + np.log(np.linalg.det(cov_d_s)))



    ### Find the nodes with maximum loglikelihood and return the nodes
    # with maximum a posteriori likelihood
    ### Corrects a bias

    scores = sorted(loglikelihood.items(), key=operator.itemgetter(1), reverse=True)
    source_candidate = scores[0][0]

    return source_candidate, scores

#################################################### Helper methods for ml algo
def posterior_from_logLH(loglikelihood):
    """Computes and correct the bias associated with the loglikelihood operation.
    The output is a likelihood.

    Returns a dictionary: node -> posterior probability

    """
    bias = logsumexp(list(loglikelihood.values()))
    return dict((key, np.exp(value - bias))
            for key, value in loglikelihood.items())
