import re
import os
import sys
import time as tm
import random

sys.path.append("..")

import numpy as np

from network import LPNetwork, InfeasibleProblemException


class OvercontractException(Exception):
    pass


def AddRandomDemands(net: LPNetwork, numberDemands: int, flow: int = 1):
    """
    Function that adds random demands to a network
    """
    nodes = [i for i in net.graph.nodes]
    for i in range(numberDemands):
        src, dst = np.random.choice(nodes, replace=False, size=(2,))
        if type(src) is np.int64:
            net.AddContract(int(src), int(dst), int((flow)), name=f"{i}")
        else:
            net.AddContract(src, dst, int(flow))


with open("graph-list.txt") as fil:
    graphs = fil.read().split("\n")

numContractsList = [100]
FLOW = 1

for graphName in graphs:
    for numContracts in numContractsList:
        attempts = 0
        print(f"{graphName} - {numContracts}")
        while True:
            if attempts > 50:
                break
            graph = LPNetwork("./blank_graphs/" + graphName)
            AddRandomDemands(graph, numContracts, FLOW)
            try:
                print(f"({attempts}) Solving LP.")
                attempts += 1
                execTime = -tm.process_time()
                graph.SolveLP()
                execTime += tm.process_time()
            except InfeasibleProblemException:
                print("Failed. Infeasible problem. recreating...")
                continue
            print("Success")
            resultName = re.sub(r"no-contracts", f"{numContracts}-contracts", graphName)
            # Save LP results in graph
            graph.graph.graph["lp_exec_time"] = graph.GetLastLPExecutionTime()
            graph.graph.graph["lp_prep_time"] = graph.GetLPPreparationTime()
            graph.graph.graph["lp_optimal_cost"] = graph.GetLPOptimalCost()
            # Saves graph as json
            graph.SaveGraph(f"{FLOW}-flow-" + resultName)
            break
