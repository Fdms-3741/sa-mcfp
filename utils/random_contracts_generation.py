#
# random_contracts_generation
# Author: Fernando Dias
#
# Script that receives a network and iteratively adds contracts and solves via LP in order to obtain the maximum number of contracts for a given network
#
import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from network import LPNetwork, InfeasibleProblemException

networkName = "rnp"
defaultCapacity = 100
maximumNumberContracts = 1000
contractsPerLoop = 50

a = LPNetwork(networkName)

for link in a.graph.edges:
    a.graph.edges[link]["capacity"] = defaultCapacity

nodes = list(a.graph.nodes)

times = {}
numContracts = 0
for i in range(maximumNumberContracts // contractsPerLoop):
    for j in range(contractsPerLoop):
        source, destination = np.random.choice(nodes, size=(2,), replace=False)
        a.AddContract(source, destination, 2, name=f"{contractsPerLoop*i+j}")
        numContracts += 1
    try:
        print(f"Solving for {numContracts} contracts...")
        a.SolveLP()
        times[numContracts] = a.GetLastLPExecutionTime()
    except InfeasibleProblemException or KeyboardInterrupt:
        for k in range(20):
            a.RemoveContract(f"{20*i+j-k}")
        a.SaveGraph(f"{networkName}_with_{numContracts}_contracts.json")
        break

print(f"Finished loop with {numContracts} added contracts to {networkName} network")
a = pd.Series(times)
a.name = "Elapsed time"
a.index.name = "Number of contracts"

a.to_csv("elapsed_time_by_number_of_contracts.csv")
