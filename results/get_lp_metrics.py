import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append("..")

from network import LPNetwork, SANetwork

baseDir = "../network_models/test_graphs/"

data = []
for i in os.listdir(baseDir):
    network = re.search(r"(geant|rnp|random)", i)
    if not network:
        continue
    numContracts = re.search(r"(\d+)-contracts", i)
    if not numContracts:
        continue

    lpNet = LPNetwork(baseDir + i)
    saNet = SANetwork(baseDir + i)

    data.append(
        {
            "Network name": network.group(1),
            "Number of contracts": len(lpNet.ListContracts()),
            "Cost range": (int(np.min(saNet.PropertyVector('weight'))),int(np.max(saNet.PropertyVector('weight')))),
            "Capacity range": (int(np.min(saNet.PropertyVector('capacity'))),int(np.max(saNet.PropertyVector('capacity')))),
            "Optimal cost": int(lpNet.graph.graph["lp_optimal_cost"]),
        }
    )

data = pd.DataFrame(data)
print(data)
data = data.style.format(precision=2)
data.to_latex("lp_metrics.tex")
