import os
os.chdir('../')
import networkx as nx
import test_source_loc as ts
import scipy.stats as st
import random
from datetime import datetime
random.seed(random.random())


graph=nx.path_graph(20)
dist=st.expon()
source=10
infected = nx.single_source_dijkstra_path_length(graph, source)
obs=[0,19]
obs_time = dict((k,v) for k,v in infected.items() if k in obs)
res=ts.gau_li_empirical_t0(graph, obs_time, dist)[1]
rank=-1
for i in range(20):
    if res[i][0]==source:
        rank=i
print("Path on 20 nodes, source rank ",rank)


graph=nx.erdos_renyi_graph(20,0.5)
dist=st.expon()
source=10
infected = nx.single_source_dijkstra_path_length(graph, source)
obs=[0,19]
obs_time = dict((k,v) for k,v in infected.items() if k in obs)

res=ts.gau_li_empirical_t0(graph, obs_time, dist)[1]
rank=-1
for i in range(20):
    if res[i][0]==source:
        rank=i
print("G(20,0.5), source rank ",rank)

