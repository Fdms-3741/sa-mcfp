import os
import re
import sys

import pandas as pd
import numpy as np

sys.path.append("..")
from network import SANetwork
from simulated_annealing.graph_solutions import MinCostSolution

path = "../network_models/test_graphs/"
data = {}

for graph in os.listdir(path):
    completeName = re.search(
        r"(geant|rnp|random)(-(\d{2})-(\d{2,3}))?-(\d{3,4})-contracts", graph
    )
    name, _, nodes, links, contracts = completeName.groups()

    # Se nome == rnp ou geant, não definido nodes e links
    print(name, nodes, links, contracts)

    a = SANetwork(path + graph)
    sol = MinCostSolution(a)
    freeVars = sol.x.shape[1]

    data[(name, nodes, links, contracts)] = freeVars

a = pd.Series(data)
a.to_pickle("free_variables.pickle")
print(a)
