# solucoes.py
#
# Arquivo que contém as classes de solução para o problema dos graphs
#
import time as tm

from networkx.algorithms import enumerate_all_cliques
import numpy as np
import networkx as nx
import pandas as pd
from pandas.core.reshape.merge import warnings

from .sa import Solution
import sys

sys.path.append("../..")
from network import SANetwork

rng = np.random.default_rng()


class GraphSolution(Solution):
    r"""
    This class implements a :method:`simulated_annealing.simulated_annealing.Solution` for the
    flows of each link for each contract in the :class:`network.SANetwork`.

    This creates the elements (:math:`x`, :math:`x_{\min}` and :math:`\hat{x}`) of the solution
    space and the candidate function.

    The cost functions are implemented by the following derived classes:

    .. inheritance-diagram:: simulated_annealing.solutions.ViableSolution simulated_annealing.solutions.MinCostSolution simulated_annealing.solutions.BalanedSolution


    """

    def SwitchCandidate(self, candidate="IncDec"):
        self._currentCandidate = candidate

    def Candidate(self, epsilon):
        if not hasattr(self, "_currentCandidate"):
            raise Exception("Current candidate function not set")

        self.candidateDelay -= tm.perf_counter()
        if self._currentCandidate == "IncDec":
            self.CandidateIncrementDecrement(epsilon)
        elif self._currentCandidate == "AddNormalVectorAnswer":
            self.AddNormalVectorAnswer(epsilon)
        elif self._currentCandidate == "AddNormalVectorContract":
            self.AddNormalVectorContract(epsilon)
        elif self._currentCandidate == "AddNormalVectorLink":
            self.AddNormalVectorLink(epsilon)
        elif self._currentCandidate == "AddNormalValue":
            self.AddNormalValue(epsilon)
        elif self._currentCandidate == "AddCauchyVectorAnswer":
            self.AddCauchyVectorAnswer(epsilon)
        elif self._currentCandidate == "AddCauchyVectorContract":
            self.AddCauchyVectorContract(epsilon)
        elif self._currentCandidate == "AddCauchyVectorLink":
            self.AddCauchyVectorLink(epsilon)
        elif self._currentCandidate == "AddCauchyValue":
            self.AddCauchyValue(epsilon)
        else:
            raise NotImplementedError("Candidate type not implemented")
        self.candidateDelay += tm.perf_counter()
        self.candidateDelayMeasures += 1

    def CandidateIncrementDecrement(self, epsilon):
        # Creates a mask with only
        mask = np.zeros(self.x.shape).ravel()
        numChanges = min(int(np.random.exponential(epsilon) + 1), mask.shape[0] - 1)
        # Sorts <numChanges> indexes and adds either -1 or 1 to value
        mask[
            np.random.choice(np.arange(mask.shape[0]), size=numChanges, replace=False),
        ] = np.random.choice([-1, 1], size=numChanges)
        # Reshapes to original x shape
        mask = mask.reshape(self.x.shape)
        # Adds to result
        self.xhat = self.x + mask

    def AddCauchyValue(self, epsilon):
        self.xhat = self.x
        selectedContract = np.random.randint(self.x.shape[0])
        selectedDirection = np.random.randint(self.x.shape[1])
        self.xhat[selectedContract, selectedDirection] += np.round(
            np.random.standard_cauchy() * epsilon
        ).astype("int")

    def AddCauchyVectorLink(self, epsilon):
        self.xhat = self.x
        selectedContract = np.random.randint(self.x.shape[0])
        self.xhat[selectedContract, :] += np.round(
            epsilon * np.random.standard_cauchy(size=(self.x.shape[1]))
        ).astype("int")

    def AddCauchyVectorContract(self, epsilon):
        self.xhat = self.x
        selectedLink = np.random.randint(self.x.shape[1])
        self.xhat[:, selectedLink] += np.round(
            epsilon * np.random.standard_cauchy(size=(self.x.shape[0]))
        ).astype("int")

    def AddCauchyVectorAnswer(self, epsilon):
        self.xhat = self.x
        self.xhat += np.round(
            epsilon * np.random.standard_cauchy(size=self.x.shape)
        ).astype("int")

    def AddNormalVectorLink(self, epsilon):
        self.xhat = self.x
        selectedContract = np.random.randint(self.x.shape[0])
        self.xhat[selectedContract, :] += np.round(
            np.random.normal(0, epsilon, size=(self.x.shape[1]))
        ).astype("int")

    def AddNormalVectorContract(self, epsilon):
        self.xhat = self.x
        selectedLink = np.random.randint(self.x.shape[1])
        self.xhat[:, selectedLink] += np.round(
            np.random.normal(0, epsilon, size=(self.x.shape[0]))
        ).astype("int")

    def AddNormalVectorAnswer(self, epsilon):
        self.xhat = self.x
        if epsilon >= 1 or epsilon <= 0:
            raise Exception("Invalid use of epsilon for particular candidate")
        mask = np.zeros(self.x.shape).ravel().astype("int64")
        numChanges = int(np.ceil(epsilon * mask.shape[0]))
        mask[
            np.random.choice(np.arange(mask.shape[0]), size=numChanges, replace=False)
        ] = np.round(np.random.normal(0, 1, size=numChanges)).astype("int")
        self.xhat += mask.reshape(self.xhat.shape)

    def AddNormalValue(self, epsilon):
        self.xhat = self.x
        selectedContract = np.random.randint(self.x.shape[0])
        selectedDirection = np.random.randint(self.x.shape[1])
        self.xhat[selectedContract, selectedDirection] += np.round(
            np.random.normal(0, 1 * epsilon)
        ).astype("int")


class NullspaceSolution(GraphSolution):
    def __init__(self, graph: SANetwork, x0="random"):
        self.graph = graph
        self.costDelay = 0
        self.costDelayMeasures = 0
        self.candidateDelay = 0
        self.candidateDelayMeasures = 0
        self.graph.CalculateNullspaceConversionMatrix()
        self.FlowCalculation = self.graph.FlowCalculationFunction()
        self._capacityVector = self.graph.PropertyVector("capacity")
        self._costVector = self.graph.PropertyVector("weight")

        if x0 == "random":
            self.x = np.random.randint(-25, 25, size=self.graph.solutionShape).astype(
                "int64"
            )
        elif x0 == "zeros":
            self.x = np.zeros(self.graph.solutionShape)
        elif x0 == "sp":
            self.x = np.zeros(self.graph.solutionShape)
            self.StartSmallestPath()
        else:
            raise NotImplementedError(f"Initialization {x0} for x not implemented")

        self.xhat = self.x
        self.xmin = self.x

        self.stopAlgorithm = False

    def StartSmallestPath(self):
        nullspaceMatrix = self.graph.CalculateNullspaceConversionMatrix()[0][0, :, :-1]
        # Calculates the inverse and rounds up to the nearest integer
        # TODO: Find better pseudo-inverse definition that doesn't return orthonormal vectors
        # Those vectors tend to have smaller values that get truncated in the rounding up to nearest integer
        invFlowMatrix = np.linalg.pinv(nullspaceMatrix)

        # For each demand, calculates the shortest path and
        # uses the inverse matrix to find the nearest representation
        demands = self.graph.ListContracts()
        for idxDemand, demand in enumerate(demands):
            contract = self.graph.GetContract(demand)

            pathNodes = nx.shortest_path(
                nx.Graph(self.graph.graph), contract["source"], contract["destination"]
            )

            pathEdges = [(i, j) for i, j in zip(pathNodes[:-1], pathNodes[1:])]
            # In case the edge is defined in the oposite direction
            pathEdges += [(j, i) for i, j in zip(pathNodes[:-1], pathNodes[1:])]

            flows = np.zeros((len(self.graph.graph.edges), 1))
            for idxFlow, edge in enumerate(self.graph.graph.edges):
                # Check if current edge is in the list generated for the path
                if edge in pathEdges:
                    flows[idxFlow, 0] = 1

            self._x[idxDemand, :] = np.round(invFlowMatrix @ flows).astype("int")[:, 0]
            self.x = self.x

    def CapacityConstraintPenalty(self, x):
        excess = (
            np.sum(np.abs(self.FlowCalculation(x)), axis=0) - self._capacityVector[:, 0]
        )
        self.excess = np.sum(np.where(excess > 1e-5, excess, 0))
        return self.excess


class ViableSolution(NullspaceSolution):
    r"""
    The ViableSolution space consists of any answers that satisfy the restrictions. Thus, any valid
    answer is 0 and any value greater than 0 is generated by at least one node not following the restrictions.

    The viable solution cost is calculated as follows:

    .. math::
        J(\mathbb{x}_{ij}) = \sum_j(|\sum_{i}\mathbb{x}-\mathbb{c}_j > 0)^2

    Where :math:`i` indexes all of the commodities and :math:`j` indexes all links in the network.
    """

    def _Cost(self, x):
        x = self.FlowCalculation(x)  # [contratos][enlaces]
        self.CapacityConstraintPenalty(x)
        if np.sum(self.excess) < 1e-10:
            self.stopCondition = "Solução aceita"
            self.stopAlgorithm = True
        return np.sum(self.excess)

    def _IsViable(self, x):
        return self.jmin < 1e-5


class MinCostNormSolution(NullspaceSolution):
    """
    This class makes the SA algorithm find the minimum cost solution for the MCFP.
    Its cost function consist on the sum of the weights by their
    """

    def _Cost(self, x) -> float:
        flow = self.FlowCalculation(x)
        self.CapacityConstraintPenalty(flow)
        return (
            np.sum(
                np.sum(np.abs(flow), axis=0)
                * (self._costVector[:, 0] / np.mean(self._costVector))
            )
            + 1e3 * self.excess
        )

    def _IsViable(self, x):
        return self.excess == 0

    def GetMinimumCost(self):
        return np.sum(
            np.sum(np.abs(self.FlowCalculation(self.xmin)), axis=0)
            * self._costVector[:, 0]
        )


class MinCostSolution(NullspaceSolution):
    """
    This class makes the SA algorithm find the minimum cost solution for the MCFP.
    Its cost function consist on the sum of the weights by their
    """

    def _Cost(self, x) -> float:
        self.costDelay -= tm.perf_counter()
        flow = self.FlowCalculation(x)
        self.CapacityConstraintPenalty(x)
        res = np.sqrt(
            np.sum(np.sum(np.abs(flow), axis=0) * (self._costVector[:, 0]))
            + 1e4 * x.shape[1] * self.excess
        )
        self.costDelay += tm.perf_counter()
        self.costDelayMeasures += 1
        return res

    def _IsViable(self, x):
        return self.excess == 0

    def GetMinimumCost(self):
        flow = self.FlowCalculation(self.xmin)
        return (
            np.sum(np.sum(np.abs(flow), axis=0) * (self._costVector[:, 0]))
            + 1e4 * self.xmin.shape[1] * self.excess
        )


class QueueDelayMinimization(NullspaceSolution):
    """
    Class that implements the minimum cost solution by calculating K-paths
    between each flow and assigning it a flow value.
    """

    def _Cost(self, x) -> float:
        self.costDelay -= tm.perf_counter()
        flow = np.abs(self.FlowCalculation(x))
        if (excess := self.CapacityConstraintPenalty(x)) > 0:
            result = np.multiply(*flow.shape) * 10 + excess
        else:
            flowSum = np.sum(flow, axis=0)[:, np.newaxis]
            linkWeights = 1 / (
                self._capacityVector
                - np.where(
                    flowSum < self._capacityVector, flowSum, self._capacityVector - 1
                )
            )
            result = np.sum(flow @ linkWeights) + 1e5 * self.CapacityConstraintPenalty(
                x
            )
        self.costDelay += tm.perf_counter()
        self.costDelayMeasures += 1
        return float(result)

    def GetMinimumCost(self):
        return self._Cost(self.xmin)

    def _IsViable(self, x):
        return self.CapacityConstraintPenalty(x) == 0


class BalancedSolution(NullspaceSolution):
    """
    Class that implements the balanced solution for the simulated annealing
    """

    def _Cost(self, x):
        x = self.FlowCalculation(x)
        self.CapacityConstraintPenalty(x)

        return np.sum(
            np.power((np.sum(np.abs(x), axis=0) / self._capacityVector), 2)
        ) + 1e6 * np.sum(self.excess)

    def _IsViable(self, x):
        return np.sum(self.excess) < 1e-5


class KPathsSolution(GraphSolution):
    def __init__(self, graph: SANetwork, kpaths=5):
        self.graph = graph
        self.costDelay = 0
        self.costDelayMeasures = 0
        self.candidateDelay = 0
        self.candidateDelayMeasures = 0
        # Creates and populates a path link matrix
        # Starts with ones so the cost of unexisting paths is the highest possible
        self.pathLinkMatrix = np.ones(
            (len(self.graph.ListContracts()), kpaths, len(self.graph.graph.edges))
        )
        self.demandVector = np.zeros(len(self.graph.ListContracts()))
        for numContract, contract in enumerate(self.graph.ListContracts()):
            curPath = 0
            contractData = self.graph.GetContract(contract)
            source, destination = contractData["source"], contractData["destination"]
            self.demandVector[numContract] = contractData["demand"]
            for numPath, path in enumerate(
                nx.shortest_simple_paths(
                    nx.Graph(self.graph.graph), source, destination
                )
            ):
                pathEdgesFwd = []
                pathEdgesBwd = []
                for i, j in zip(path[:-1], path[1:]):
                    pathEdgesFwd.append((i, j))
                    pathEdgesBwd.append((j, i))

                pathLinkFwd = pd.Series(
                    [1 if (i in pathEdgesFwd) else 0 for i in self.graph.graph.edges]
                )
                pathLinkBwd = pd.Series(
                    [-1 if (i in pathEdgesBwd) else 0 for i in self.graph.graph.edges]
                )

                self.pathLinkMatrix[numContract][numPath] = pathLinkFwd + pathLinkBwd

                assert np.sum(np.abs(self.pathLinkMatrix[numContract][numPath])) == (
                    len(path) - 1
                )

                curPath += 1
                if curPath == kpaths:
                    break

        # Adds edges weights as cost
        self.costMatrix = self.graph.PropertyVector("weight")

        self.capacityVector = self.graph.PropertyVector("capacity")[:, 0]

        # Variable definition: Flow value for the n-th flow and the m-th possible path of that flow
        difference = self.demandVector
        correctionMatrix = np.concatenate(
            [
                difference[:, np.newaxis],
                np.zeros((self.demandVector.shape[0], kpaths - 1)),
            ],
            axis=1,
        )
        correctionMatrix = self.shuffle_along_axis(correctionMatrix, 1)
        self.x = correctionMatrix[:, :, np.newaxis]
        self.xmin = self.x
        self.xhat = self.x
        self.stopAlgorithm = False

    def shuffle_along_axis(self, a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    def SwitchSum(self, epsilon):
        contracts = np.random.choice(
            self.x.shape[0], size=np.random.randint(1, epsilon + 1), replace=True
        )
        self.xhat = self.x

        for contract in contracts:
            add = np.random.choice(
                np.arange(self.x.shape[1])[
                    self.x[contract, :, 0] < self.demandVector[contract]
                ]
            )
            sub = np.random.choice(
                np.arange(self.x.shape[1])[self.x[contract, :, 0] > 0]
            )
            self.xhat[contract][add][0] += 1
            self.xhat[contract][sub][0] -= 1

        assert (np.sum(self.x, axis=1) == self.demandVector).all()

    def Candidate(self, epsilon):
        self.candidateDelay -= tm.perf_counter()
        self.SwitchSum(epsilon)
        self.candidateDelay += tm.perf_counter()
        self.candidateDelayMeasures += 1

    def CorrectCandidate(self, x):
        """
        Corrects the flow of the candidate solution so the sum of the flows for a demand always matches the requested demand.
        """
        # Corrects for a negative flow
        x = np.where(x < 0, -x, x)
        # Corrects for flow distribution
        totalFlows = np.sum(x, axis=1)
        difference = self.demandVector - totalFlows
        for curContract, totalDiff in enumerate(difference):
            currentDiff = totalDiff
            while True:
                if currentDiff == 0:
                    break
                path = np.random.randint(0, x.shape[1])
                if currentDiff > 0:  # Sum to flow
                    x[curContract][path] += 1
                    currentDiff -= 1
                if currentDiff < 0 and x[curContract][path] > 0:
                    currentDiff += 1
                    x[curContract][path] -= 1

        assert (
            np.sum(x, axis=1) == self.demandVector
        ).all(), f"Unexpected behaviour for {x}"

        self.xhat = x

    def FlowCalculation(self, x):
        return np.sum(x * np.abs(self.pathLinkMatrix), axis=1)

    def DirectedFlowCalculation(self, x):
        return np.sum(x[:, :, np.newaxis] * self.pathLinkMatrix, axis=1)

    def _CapacityConstraintPenalty(self, x) -> float:
        """
        Returns a non-negative number proportional to whether the current solution x
        contains any flow ABOVE a certain link's capacity
        """
        totalLinkFlow = np.sum(self.FlowCalculation(x), axis=0)

        overflowCapacity = (totalLinkFlow) * ((totalLinkFlow - self.capacityVector) > 0)

        self.capacityConstraintPenalty = float(np.linalg.norm(overflowCapacity))

        return self.capacityConstraintPenalty

    def _CapacityReachedPenalty(self, x) -> float:
        """
        Returns a results for whether the current solution x contains any flow that is
        EQUAL OR ABOVE a certain link's capacity
        """
        totalLinkFlow = np.sum(self.FlowCalculation(x), axis=0)

        overflowCapacity = (totalLinkFlow + 1) * (
            (totalLinkFlow - self.capacityVector) >= 0
        )

        self.capacityReachedPenalty = float(np.linalg.norm(overflowCapacity))

        return self.capacityReachedPenalty


class KPathsMinimizationNorm(KPathsSolution):
    """
    Class that implements the minimum cost solution by calculating K-paths
    between each flow and assigning it a flow value.
    """

    def _Cost(self, x) -> float:
        result = np.sum(
            self.FlowCalculation(x) @ (self.costMatrix / np.mean(self.costMatrix))
        ) + 10000 * self._CapacityConstraintPenalty(x)

        return np.sqrt(result)

    def GetMinimumCost(self):
        return np.sum(
            self.FlowCalculation(self.xmin) @ self.costMatrix
        ) + 10000 * self._CapacityConstraintPenalty(self.xmin)

    def _IsViable(self, x):
        return self.capacityConstraintPenalty == 0


class KPathsQueueDelayMinimization(KPathsSolution):
    """
    Class that implements the minimum cost solution by calculating K-paths
    between each flow and assigning it a flow value.
    """

    def _Cost(self, x) -> float:
        self.costDelay -= tm.perf_counter()
        flow = np.abs(self.FlowCalculation(x))
        if (excess := self._CapacityConstraintPenalty(x)) > 0:
            result = np.multiply(*flow.shape) * 10 + excess
        else:
            flowSum = np.sum(flow, axis=0)[:, np.newaxis]
            linkWeights = 1 / (
                self.capacityVector
                - np.where(
                    flowSum < self.capacityVector, flowSum, self.capacityVector - 1
                )
            )
            result = np.sum(flow @ linkWeights) + 1e5 * self._CapacityConstraintPenalty(
                x
            )
        self.costDelay += tm.perf_counter()
        self.costDelayMeasures += 1
        return float(result)

    def GetMinimumCost(self):
        return self._Cost(self.xmin)

    def _IsViable(self, x):
        return self._CapacityConstraintPenalty(x) == 0


class KPathsMinimization(KPathsSolution):
    """
    Class that implements the minimum cost solution by calculating K-paths
    between each flow and assigning it a flow value.
    """

    def _Cost(self, x) -> float:
        self.costDelay -= tm.perf_counter()
        result = np.sum(
            self.FlowCalculation(x) @ self.costMatrix
        ) + 10000 * self._CapacityConstraintPenalty(x)
        self.costDelay += tm.perf_counter()
        self.costDelayMeasures += 1
        return np.sqrt(result)

    def GetMinimumCost(self):
        return np.sum(
            self.FlowCalculation(self.xmin) @ self.costMatrix
        ) + 10000 * self._CapacityConstraintPenalty(self.xmin)

    def _IsViable(self, x):
        return self._CapacityConstraintPenalty(x) == 0


class KPathsViable(KPathsMinimization):
    def _Cost(self, x) -> float:
        return self._CapacityConstraintPenalty(x)
