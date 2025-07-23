import numpy as np
import scipy as sp
import sympy as sym
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from network import SANetwork, LPNetwork

from simulated_annealing.sa import SimulatedAnnealing
from simulated_annealing.graph_solutions import (
    MinCostSolution,
    KPathsMinimization,
    MinCostNormSolution,
)

netName = "./network_models/test_graphs/1-flow-rnp-100-contracts.json"
saNet = SANetwork(netName)
lpNet = LPNetwork(netName)

F, D = saNet.CalculateFlowConservationEquation()

sol = MinCostNormSolution(saNet)

sol.SwitchCandidate("AddNormalVectorAnswer")
sa = SimulatedAnnealing(sol, 1000, 20, 50, 0.5, saveStep=50)

sa.Execute()
