import os
import sys

import pandas as pd
import numpy as np
import networkx as nx

sys.path.append("..")
from network import SANetwork


def GetAverageNumberOfPaths(graph):
    nodes = list(graph.nodes)
    totalPaths = 0
    totalPairs = 0
    for srcIdx in range(len(nodes)):
        for dstIdx in range(srcIdx + 1, len(nodes)):
            totalPairs += 1
            src = nodes[srcIdx]
            dst = nodes[dstIdx]
            totalPaths += len(list(nx.all_simple_paths(graph, src, dst)))
    return totalPaths / totalPairs


def GetStats(graph: nx.Graph):
    res = pd.Series()
    res["Node amount"] = len(graph.nodes)
    res["Edges amount"] = len(graph.edges)
    res["Edge to node ratio"] = len(graph.edges) / len(graph.nodes)
    res["Average shortest path length"] = nx.average_shortest_path_length(graph)
    # res['Average number of paths'] = GetAverageNumberOfPaths(graph)
    res["Average node connectivity"] = nx.average_node_connectivity(graph)
    res["Average node normalized load centrality"] = np.mean(
        [j for _, j in nx.load_centrality(graph, normalized=True).items()]
    )
    return res


path = "../network_models/blank_graphs/"
data = {}
for graph in os.listdir(path):
    if graph[:3] == "rnp":
        name = "_rnp"
    elif graph[:3] == "gea":
        name = "_geant"
    else:
        name = graph[:12]
    print(graph)
    data[name] = GetStats(nx.Graph(SANetwork(path + graph).graph))

a = pd.DataFrame(data)
a = a.T.sort_index().T
a.to_latex("network_metrics.tex")
print(a)
