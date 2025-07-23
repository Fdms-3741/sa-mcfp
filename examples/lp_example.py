import sys
from network import LPNetwork

if len(sys.argv) > 1:
    graphName = sys.argv[1]
else:
    graphName = "./network_models/random-20-60-100-contracts.json"

graph = LPNetwork(graphName)

graph.SolveLP()

graph.SaveGraph(graphName)

print(graph.GetLastLPExecutionTime())
print(graph.GetLPOptimalCost())
