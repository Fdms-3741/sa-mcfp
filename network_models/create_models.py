import sys
import random

sys.path.append("..")

import networkx as nx
import matplotlib.pyplot as plt

from network import SANetwork
from utils.topologyGeneration import TopologyGenerator


def CreateRandomConnectedGraph(nodes, links, weightList, capacityList):
    currentAttempt = 0

    a = SANetwork(None)

    # Create nodes
    for i in range(nodes):
        a.AddNode(i)

    # Create a ring chain
    for i in range(nodes):
        a.AddLink(
            i, (i + 1) % nodes, random.choice(weightList), random.choice(capacityList)
        )

    # Create links
    nodesList = list(a.graph.nodes)
    for _ in range(links - nodes):
        while True:
            src, dst = random.choice(nodesList), random.choice(nodesList)
            # Check if link already exists before exiting loop
            if src != dst and a.graph.get_edge_data(src, dst, None) is None:
                break
        a.AddLink(src, dst, random.choice(weightList), random.choice(capacityList))

    return a


def CreateTestbenchGraph(nodes, links, capacityList, weightList):
    net = CreateRandomConnectedGraph(nodes, links, weightList, capacityList)
    print("Nodes:", len(net.graph.nodes))
    print("Links:", len(net.graph.edges))
    net.SaveGraph(f"./blank_graphs/random-{nodes}-{links}-no-contracts.json")


if __name__ == "__main__":
    CreateTestbenchGraph(60, 100, list(range(100, 1000, 100)), list(range(1, 11, 2)))
