import json
import time

import numpy as np

# import scipy as sp
import sympy as sym

# import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pulp
from pulp import PULP_CBC_CMD

__DEBUG__ = True


class Network:
    """
    General class that represents a network.

    This class is used to define a network and their respective


    .. inheritance-diagram:: network.SANetwork network.LPNetwork
    """

    def __init__(self, graph, defaultWeight=1, defaultCapacity=10):
        """
        Class constructor

        :param str graph: Can be of type nx.Graph(), nx.DiGraph(), or a string containing an .json from exported graph or csv of edge list
        :param defaultWeight: Default weight to be used on edge list
        :type defaultWeight: int or None
        :param defaultCapacity: Default capacity to be used on edge list
        :type defaultCapacity: int or None

        """
        self.graph: nx.DiGraph
        if type(graph) is str:
            if graph[-5:] == ".json":
                with open(graph) as fil:
                    data = json.load(fil)
                self.graph = nx.node_link_graph(data, edges="links")
            elif graph[-4:] == ".csv":
                with open(graph) as fil:
                    data = fil.read().split("\n")
                self.graph = nx.parse_edgelist(
                    data, delimiter=",", create_using=nx.DiGraph
                )
                self.AddDefaultValuesNewGraph(defaultWeight, defaultCapacity)
            else:
                raise Exception("Invalid file name")
        elif graph is None:
            self.graph = nx.DiGraph()
        elif type(graph) is nx.Graph or type(graph) is nx.DiGraph:
            self.graph = nx.DiGraph(graph)
        else:
            raise Exception("Untreated type")

        self.lp_time = None

    def AddDefaultValuesNewGraph(self, defaultWeight, defaultCapacity):
        for link in self.graph.edges:
            self.graph.edges[link]["capacity"] = defaultCapacity
            self.graph.edges[link]["weight"] = defaultWeight

    def FindNodeIndex(self, node):
        """
        For a given node name, it finds its index in the list of added nodes

        :param str node: Node name

        :return int: Node index
        """
        for i, n in enumerate(self.graph.nodes):
            if n == node:
                return i
        return None

    def AddLinkProperty(self, name, values):
        """
        Adds a property value for all links in the network.

        :param str name: Property's name
        :param list(float) values: List of values for this property for each link. This list must match the number of links in the graph.
        """
        if len(values) != len(self.graph.edges):
            raise Exception(
                "Values must have the same number of elements as the graph has edges"
            )

        for idx, edge in enumerate(self.graph.edges):
            self.graph.edges[edge][name] = values[idx]

    def GetNodeList(self):
        """
        Returns a list of added nodes

        :return list: List of nodes
        """
        return list(self.graph.nodes)

    def GetLinksList(self):
        """
        Returns a list of links between nodes as a (source,destination) tuple
        """
        return list(self.graph.edges)

    def PlotGraph(
        self,
        property=None,
        contract=None,
        layout=None,
        showLabels=True,
        nodeSize=500,
    ):
        r"""
        Plots the current graph

        :param str property: (default None) Link property's name to display in each link
        :param str contract: (default None) Contract's name to highlight the start and end nodes
        :param str layout: Type of node disposition

        Layout can be one of the following alternatives:

            * ``'circular'``: Places all nodes within the perimeter of a circle
            * ``'spring'``: Uses the "spring" algorithm from networkx for node disposition
            * ``'position'``: Reads a ``pos`` property of each node and places in a graph based on that property. ``pos`` must be defined for every node and must be a tuple with ``(x,y)`` coordinates.

        """
        if not layout or layout not in ["circular", "spring", "position"]:
            layout = "circular"

        if layout == "spring":
            pos = nx.spring_layout(self.graph, weight="weight")
        elif layout == "position":
            pos = nx.get_node_attributes(self.graph, "pos")
        else:
            pos = nx.circular_layout(self.graph)

        # Draw nodes
        nodeColors = ["cyan"] * len(self.graph.nodes)
        if contract:
            if contract not in self.graph.graph["contracts"]:
                raise Exception("O contract não existe no graph")
            property = contract
            # Mudar a cor se a property for um contract
            for idx, nodeName in enumerate(self.graph.nodes):
                if (
                    "contracts" in self.graph.nodes[nodeName].keys()
                    and contract in self.graph.nodes[nodeName]["contracts"].keys()
                ):
                    nodeColors[idx] = "red"

        nx.draw_networkx_nodes(
            self.graph, pos, node_color=nodeColors, node_size=nodeSize
        )
        # Draw edges
        # Draw edge labels
        edgeColors = ["black"] * len(self.graph.edges)
        if property:
            edge_labels = nx.get_edge_attributes(self.graph, property)
            # Se contract, pinta de cinza os links que não tem o flow de um contract
        #            print(edge_labels)
        #            if contract:
        #                for idx, (edge, flow) in enumerate(edge_labels.items()):
        #                    if np.abs(flow) < 1e-5:
        #                       edgeColors[idx] = 'grey'
        else:
            edge_labels = {}
            for idx, edge in enumerate(self.graph.edges):
                edge_labels[edge] = idx

        nx.draw_networkx_edges(self.graph, pos, edge_color=edgeColors, arrows=True)

        for key, item in edge_labels.items():
            edge_labels[key] = f"{item:.2f}"

        if showLabels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        # Draw node labels
        if showLabels:
            nx.draw_networkx_labels(self.graph, pos)
        # Show plot
        plt.title(f"{property} values")
        plt.axis("off")

        return plt.gcf(), plt.gca()

    def ListContracts(self):
        """
        Returns a list of all the contracts added to the network
        """
        if "contracts" not in self.graph.graph.keys():
            return []
        return self.graph.graph["contracts"].keys()

    def AddNode(self, name, **kwargs):
        self.graph.add_node(name, contracts={}, **kwargs)

    def RemoveNode(self, name):
        self.graph.remove_node(name)

    def AddLink(self, source, destination, cost=1, capacity=np.inf, **kwargs):
        self.graph.add_edge(
            source, destination, weight=cost, capacity=capacity, **kwargs
        )

    def RemoveLink(self, source, destination):
        self.graph.remove_edge(source, destination)

    def SaveGraph(self, filename):
        """
        Salva o graph num arquivo json de forma que ele possa ser utilizado mais tarde
        """
        with open(filename, "w") as fil:
            json.dump(nx.node_link_data(self.graph, edges="links"), fil)

    def RemoveContract(self, name: str):
        """
        Remove um contract pelo seu name
        """
        if name not in self.graph.graph["contracts"]:
            raise Exception("contract não existente")

        source, destination = tuple(self.graph.graph["contracts"][name])

        self.graph.nodes[source]["contracts"].pop(name)
        self.graph.nodes[destination]["contracts"].pop(name)
        self.graph.graph["contracts"].pop(name)

    def AddContract(self, source, destination, flow: int, **kwargs) -> None:
        """
        Adds a new contract to the network.

        :param source: Source node
        :param destination: Destination node
        :param flow: The expected flow between source and destination.
        :param name: (Optional) Name for contract identification. Default is a string following the format `({source},{destination})`
        """
        if source not in self.graph.nodes:
            raise Exception(f"source '{source}' não existe no graph")
        if destination not in self.graph.nodes:
            raise Exception(f"destination '{destination}' não existe no graph")
        if type(flow) is not int and flow <= 0:
            raise Exception("Flow must be an integer and positive")

        if "contracts" not in self.graph.graph:
            self.graph.graph["contracts"] = {}

        if "contract_params" not in self.graph.graph:
            self.graph.graph["contract_params"] = []

        if "name" not in kwargs.keys():
            name = f"({source},{destination})"
        else:
            name = kwargs["name"]
            del kwargs["name"]

        # Adds name, source and destination to contracts dict
        self.graph.graph["contracts"][name] = [source, destination]

        # Adds for the source and destination links the contract's name and requested flow
        if "contracts" not in self.graph.nodes[source]:
            self.graph.nodes[source]["contracts"] = {}

        self.graph.nodes[source]["contracts"][name] = flow

        if "contracts" not in self.graph.nodes[destination]:
            self.graph.nodes[destination]["contracts"] = {}

        self.graph.nodes[destination]["contracts"][name] = -flow

        # Adds parameter to graph based on contract name
        for key, val in kwargs.items():
            contractParam = f"contracts_{key}"

            if contractParam not in self.graph.graph:
                self.graph.graph[contractParam] = {}

                if contractParam not in self.graph.graph["contract_params"]:
                    self.graph.graph["contract_params"].append(contractParam)

            self.graph.graph[contractParam][name] = val

    def GetContract(self, contractName):
        """
        Gets the information available about a contract based on its name

        This function returns a dict containing the name, source node name, destination node name and the parameters associated with this contract.

        The parameters are organized as a dictionary containing a name for the param and the value associated with the contract, if exists.
        If no parameter is associated with a contract, an empty dictionary is returned.



        :param contractName: The contract's name.

        :return: Dictionary containing information about a contract

        Keys for this dictionary are:

        * name
        * source
        * destination
        * parameters
        """
        if contractName not in self.graph.graph["contracts"].keys():
            raise Exception("contract não existe")

        # Extracts
        contractParams = {}
        if "contract_params" in self.graph.graph:
            for paramName in self.graph.graph["contract_params"]:
                if contractName in self.graph.graph[paramName]:
                    contractParams[paramName] = self.graph.graph[paramName][
                        contractName
                    ]
        src, dst = self.graph.graph["contracts"][contractName]
        return {
            "name": contractName,
            "source": src,
            "destination": dst,
            "demand": self.graph.nodes[src]["contracts"][contractName],
            "params": contractParams,
        }

    def CalculateFlowConservationMatrix(self):
        r"""
        Calculates the flow conservation matrix based on the current network.

        The flow conservation matrix represents the :math:`F` matrix of the linear equation :math:`F\mathbf{x}=\mathbf{d}` that dictates that for every node, the sum of the flows entering the node must equal the sum of the flows exiting the node, except in cases where the node is the source or the destination of a particular commodity.

        The function :meth:`network.Network.CalculateFlowConservationVectorBatch` calculates the `\mathbf{b}` for each contract.

        :return: A (#nodes,#edges) numpy array representing the flow conservation matrix.
        """
        # Gera a matriz de conservação de flows
        self.flowConservationMatrix = np.zeros(
            (len(self.graph.nodes), len(self.graph.edges))
        ).astype("int64")

        for idx, (source, destination) in enumerate(self.graph.edges):
            self.flowConservationMatrix[self.FindNodeIndex(source), idx] = 1
            self.flowConservationMatrix[self.FindNodeIndex(destination), idx] = -1

        return self.flowConservationMatrix

    def CalculateFlowConservationVector(self):
        r"""
        Calculates the flow conservation vector batch for all of the contracts added to the network.

        This returns a matrix with (#contracts,#nodes). The first dimension indexes the multiple :math:`\mathbf{b}` vectors from the flow conservation equation, one for each contract.

        :raises NoContractException: Raised if no contract was added before calling this function.

        :return: A (#contracts,#nodes) matrix with multiple :math:`\mathbf{b}` vectors for each added contract.
        """
        if "contracts" not in self.graph.graph.keys():
            raise NoContractException

        self.flowConservationVector = np.zeros(
            (len(self.graph.graph["contracts"]), len(self.graph.nodes))
        ).astype("int64")
        idx = 0
        for name, (source, destination) in self.graph.graph["contracts"].items():
            self.flowConservationVector[idx][self.FindNodeIndex(source)] += (
                self.graph.nodes[(source)]["contracts"][name]
            )
            self.flowConservationVector[idx][self.FindNodeIndex(destination)] += (
                self.graph.nodes[(destination)]["contracts"][name]
            )
            idx += 1

        return self.flowConservationVector

    def CalculateFlowConservationEquation(self):
        self.CalculateFlowConservationMatrix()
        self.CalculateFlowConservationVector()
        if __DEBUG__:
            print("Matriz conservação de flows")
            print(self.flowConservationMatrix)
            print("Vetor conservacao de flows")
            print(self.flowConservationVector)

        return self.flowConservationMatrix, self.flowConservationVector

    def PropertyVector(self, property=None):
        """Gera um vetor com as propertys de cada link"""
        if property is None:
            property = "_"  # Padrão para não ter property
        property = nx.get_edge_attributes(self.graph, property, default=1)
        return np.array([[property[i] for i in self.graph.edges]]).T

    def FindIndexNode(self, index):
        for idx, node in enumerate(self.graph.nodes):
            if index == idx:
                return node
        return None

    def AddFlowResults(self, flowMatrix):
        """
        Adds the flow results to each link on the network based on the flow matrix.

        The flow matrix must be a numpy array of dimensions <number-of-flows>,<number-of-edges>.

        Edges must match the order of self.ListEdges()
        """
        if len(flowMatrix.shape) != 2:
            raise Exception("Unexpected number of dimensions for flowMatrix")
        if flowMatrix.shape[0] != len(self.ListContracts()):
            raise Exception("Unexpected number of contracts for flowMatrix")
        if flowMatrix.shape[1] != len(self.graph.edges):
            raise Exception("Unexpected number of links for flowMatrix")

        for i, commodity in enumerate(self.ListContracts()):
            for j, edge in enumerate(self.graph.edges):
                self.graph.edges[edge][commodity] = flowMatrix[i][j]

    def AddFlowSumResults(self, linkMatrix):
        """
        Adds the sum result for flow in this edge
        """
        for j, edge in enumerate(self.graph.edges):
            self.graph.edges[edge]["sa_result"] = linkMatrix[j]


class LPNetwork(Network):
    """
    Class that adds fucntionality to solve the MCFP by linear programming
    """

    def __init__(self, graph, defaultWeight=1, defaultCapacity=10):
        super().__init__(graph, defaultWeight, defaultCapacity)
        self.lp_time = None
        self.optimalValue = None
        self.lp_prep_time = None

    def GetLastLPExecutionTime(self):
        """
        If :meth:`network.Network.SolveLP` method was executed previously, it returns the solver's execution time.
        """
        if self.lp_time:
            return self.lp_time
        else:
            raise OptimizationNotExecutedException

    def GetLPPreparationTime(self):
        if self.lp_prep_time:
            return self.lp_prep_time
        else:
            raise OptimizationNotExecutedException

    def GetLPOptimalCost(self):
        if self.optimalValue:
            return self.optimalValue
        else:
            raise OptimizationNotExecutedException

    def SolveLP(self, verbose=False):
        """
        Solves the current graph definition as a
        """
        self.lp_prep_time = -time.process_time()

        # Create a PuLP problem instance
        prob = pulp.LpProblem("Multicommodity_Flow_Problem", pulp.LpMinimize)

        # Create flow variables for each commodity on each edge
        flow = {}
        commodities = set(self.graph.graph["contracts"].keys())

        # Variables definition
        for contract in self.graph.graph["contracts"].keys():
            src = self.GetContract(contract)["source"]
            contractDemand = abs(self.graph.nodes[src]["contracts"][contract])
            for u, v in self.graph.edges:
                flow[(u, v, contract)] = pulp.LpVariable(
                    f"flow_{u}_{v}_{contract}",
                    cat=pulp.LpInteger,
                    lowBound=0,
                    upBound=contractDemand,
                )
                flow[(v, u, contract)] = pulp.LpVariable(
                    f"flow_{v}_{u}_{contract}",
                    cat=pulp.LpInteger,
                    lowBound=0,
                    upBound=contractDemand,
                )

        # Objective function: Minimize total cost
        prob += (
            pulp.lpSum(
                (
                    flow[(u, v, commodity)] * self.graph[u][v]["weight"]
                    + flow[(v, u, commodity)] * self.graph[u][v]["weight"]
                    for commodity in commodities
                    for u, v in self.graph.edges
                )
            ),
            "Total Cost",
        )

        # Capacity constraints
        for u, v in self.graph.edges:
            prob += (
                (
                    pulp.lpSum(
                        (
                            flow[(u, v, commodity)] + flow[(v, u, commodity)]
                            for commodity in commodities
                        )
                    )
                    <= self.graph[u][v]["capacity"]
                ),
                f"Capacity of edge ({u}, {v})",
            )

        # Flow conservation constraints
        for (
            commodity
        ) in commodities:  # For each commodity, individual restrictions apply
            # Gets the source and destination for each commodity
            source, destination = self.graph.graph["contracts"][commodity]
            # Add restrictions for every node
            for node in self.graph.nodes:
                # Flow value depends on whether the node is a souce, destination or intermediary
                flowValue = (
                    self.graph.nodes[node]["contracts"][commodity]
                    if node in (source, destination)
                    else 0
                )

                prob += (
                    (
                        pulp.lpSum(
                            (
                                flow[(node, u, commodity)]
                                for u in self.graph.successors(node)
                            )
                        )
                        - pulp.lpSum(
                            (
                                flow[(u, node, commodity)]
                                for u in self.graph.successors(node)
                            )
                        )
                        - pulp.lpSum(
                            (
                                flow[(u, node, commodity)]
                                for u in self.graph.predecessors(node)
                            )
                        )
                        + pulp.lpSum(
                            (
                                flow[(node, u, commodity)]
                                for u in self.graph.predecessors(node)
                            )
                        )
                        == flowValue
                    ),
                    f"Flow conservation for node {node} and commodity {commodity}",
                )

        if __DEBUG__:
            prob.writeLP("problem.lp")

        self.lp_prep_time += time.process_time()
        # Solve the problem
        self.lp_time = -time.process_time()
        prob.solve(PULP_CBC_CMD(msg=verbose))
        self.lp_time += time.process_time()

        problemStatus = pulp.LpStatus[prob.status]

        # Print the status of the solution
        if verbose:
            print(f"Status: {problemStatus}")

        if problemStatus != "Optimal":
            raise InfeasibleProblemException("Infeasible problem")

        self.optimalValue = prob.objective.value()

        nodesVisited = []
        flowSum = {}
        for (u, v, commodity), _ in flow.items():
            # Don't evaluate for the same edge twice
            if (u, v, commodity) in nodesVisited or (v, u, commodity) in nodesVisited:
                continue
            nodesVisited += [(u, v, commodity), (v, u, commodity)]
            # Adds the signal based on the edge's direction
            total = flow[(u, v, commodity)].value() - flow[(v, u, commodity)].value()
            # Adds to the current graph
            if (u, v) in self.graph.edges:
                self.graph.edges[(u, v)][commodity] = total
                if (u, v) not in flowSum.keys():
                    flowSum[(u, v)] = 0
                flowSum[(u, v)] += np.abs(total)
            elif (v, u) in self.graph.edges:
                self.graph.edges[(v, u)][commodity] = -total
                if (v, u) not in flowSum.keys():
                    flowSum[(v, u)] = 0
                flowSum[(v, u)] += np.abs(total)
            else:
                raise Exception(f"link {(u,v)} doesn't exist")

        for link, flow in flowSum.items():
            self.graph.edges[link]["lp_result"] = flow


class SANetwork(Network):
    """
    Adds functionality to find flow answers to the current network by returning
    the nullspace mapping for the flow conservation equations


    """

    def CalculateNullspaceConversionMatrix(self):
        """
        Calculates the answer matrix
        """
        # Calculates the flow and the demand matrices
        F = self.CalculateFlowConservationMatrix()
        b = self.CalculateFlowConservationVector()

        # Creates a symbolic flow matrix
        fMatrix = sym.matrices.Matrix(F.astype("int64"))

        # List of flows for each edge
        symbols = [sym.symbols(f"x_{i}") for i in range(F.shape[1])]

        results = []
        # Solves the Flow conservation equations for each demand
        for demand in b:
            demandVector = sym.matrices.Matrix(b.shape[1], 1, demand.astype("int64"))
            # Is the symbolic solution for a single demand
            res = sym.linsolve(fMatrix.row_join(demandVector), symbols)
            if type(res) is sym.FiniteSet:
                res = list(res)
            else:
                raise UnexpectedResultException("Unexpected result from linsolve")
            results.append(list(res)[0])

        # Creates a list of unique symbols (the free variables)
        # It's length is the kernel space size
        uniqueSymbols = list(results[0].atoms(sym.Symbol))

        # For each result, create an answer matrix that will
        # multiply the vector [freeVar_0, freeVar_1, ..., freeVar_n, 1]
        # in order to give the flow values
        #
        # To populate the matrix, associate each free variable to a column
        # and the last column to the independent term. Iterate through
        # each element of the equation to find the free variable and save
        # the constant in the appropriate cell
        #
        answerMatrix = []
        for result in results:
            answerMatrix.append(
                np.concatenate(
                    [
                        (-1) ** idx * np.array(i).astype("int64")
                        for idx, i in enumerate(
                            sym.linear_eq_to_matrix(result, *uniqueSymbols)
                        )
                    ],
                    axis=1,
                )
            )

        self.solutionConversionMatrix = np.stack(answerMatrix)
        self.numberFreeVars = len(uniqueSymbols)
        self.numberContracts = b.shape[0]
        self.solutionShape = (self.numberContracts, self.numberFreeVars)
        return self.solutionConversionMatrix, self.solutionShape

    def FlowCalculationFunction(self):
        self.CalculateNullspaceConversionMatrix()

        def CalculateFlowMatrix(sol):
            solution = np.concatenate(
                [sol, np.ones((self.numberContracts, 1)).astype("int64")], axis=1
            )
            return np.einsum(
                "ijk,ikl->ijl",
                self.solutionConversionMatrix,
                solution[:, :, np.newaxis],
            )[:, :, 0]

        return CalculateFlowMatrix

    def CalculateDemandsFromFlowMatrix(self, flowMatrix):
        return np.einsum(
            "jk,ikl->ijl",
            self.flowConservationMatrix,
            flowMatrix.copy()[:, :, np.newaxis],
        )


#
# Exceptions
#


class BaseNetworkException(Exception):
    pass


class InfeasibleProblemException(BaseNetworkException):
    pass


class NoContractException(BaseNetworkException):
    pass


class UnexpectedResultException(BaseNetworkException):
    pass


class OptimizationNotExecutedException(BaseNetworkException):
    pass
