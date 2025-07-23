import matplotlib.pyplot as plt

from network import SANetwork, LPNetwork

from simulated_annealing.sa import SimulatedAnnealing
from simulated_annealing.graph_solutions import ViableSolution, MinCostSolution

# netPath = "./network_models/real_graphs/1-flow-geant-1000-contracts.json"
netPath = "./network_models/test_graphs/test.json"

saNet = SANetwork(netPath)
lpNet = LPNetwork(netPath)

lpNet.SolveLP()

optimalSolution = lpNet.GetLPOptimalCost()

lpNet.PlotGraph("lp_result")
plt.show()

sol1 = ViableSolution(saNet)

sa1 = SimulatedAnnealing(
    sol1,
    100,
    100,
    500,
    3,
    saveStep=10,
    optimalValue=optimalSolution,
)

sa1.Execute()

print(sa1.solution.jmin)

sol2 = MinCostSolution(saNet)
sol2.x = sol1.xmin

sa2 = SimulatedAnnealing(
    sol2,
    100,
    100,
    500,
    3,
    saveStep=10,
    optimalValue=optimalSolution,
)

sa2.Execute()

print(sa2.solution.jmin)

saNet.AddFlowResults(sol2.FlowCalculation(sa2.solution.x))
