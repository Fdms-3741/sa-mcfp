#
#
# sa_solver.py: Solves MCFP using simulated annealing
# Author: Fernando Dias
#

import socket
import argparse
import time as tm
import numpy as np
from pathlib import Path
from itertools import product

import sqlalchemy
import pandas as pd
import matplotlib.pyplot as plt
from sympy.polys import NotInvertible

from network import LPNetwork, SANetwork
from simulated_annealing.graph_solutions import (
    GraphSolution,
    ViableSolution,
    MinCostSolution,
    MinCostNormSolution,
    BalancedSolution,
    KPathsMinimizationNorm,
    KPathsMinimization,
    QueueDelayMinimization,
    KPathsQueueDelayMinimization,
)
from simulated_annealing.sa import SimulatedAnnealing
from water_filling import GetSolutionWaterFilling
from save_data_sql import save_data, FindOptimal


#
# Parsing arguments
#
def ParseArguments():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Positional arguments
    parser.add_argument("graph_name", type=str, help="Name of the graph")

    #
    # Output related arguments
    #
    parser.add_argument(
        "--output", default=False, action="store_true", help="See the output"
    )
    parser.add_argument(
        "--output-id",
        default=False,
        action="store_true",
        help="Outputs the id of the results registered in the database",
    )
    parser.add_argument(
        "--progress-bar",
        default=False,
        action="store_true",
        help="See the SA's progress bar",
    )
    parser.add_argument(
        "--output-graph",
        default=False,
        action="store_true",
        help="Plot graphs related to SA",
    )
    parser.add_argument(
        "--output-network",
        default=False,
        action="store_true",
        help="Plot network diagram of current graph",
    )
    parser.add_argument(
        "--no-sql-save",
        default=False,
        action="store_true",
        help="Do not send current results to SQL database",
    )

    #
    # Simulated annealing parameters
    #
    parser.add_argument(
        "-K", type=int, default=100, nargs="+", help="Option k (default: 100)"
    )
    parser.add_argument(
        "-N", type=int, default=1000, nargs="+", help="Option n (default: 1000)"
    )
    parser.add_argument(
        "-T",
        "--T0",
        type=int,
        default=1,
        nargs="+",
        help="Option t or t0 (default: 1)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        nargs="+",
        default=1.0,
        help="Option e or epsilon (default: 1.0)",
    )
    parser.add_argument(
        "--number-of-paths",
        type=int,
        default=5,
        nargs="+",
        help="Chooses the number of paths whenever the selected solution requires specification",
    )
    parser.add_argument(
        "--save-step",
        type=int,
        default=None,
        help="Number of steps required to save internal SA metrics",
    )
    parser.add_argument(
        "--minimum-gap",
        type=float,
        default=None,
        help="Minimum gap to optimal solution",
    )
    parser.add_argument(
        "--no-lp-optimal",
        action="store_true",
        help="Do not calculate optimum value using LP if doesn't exist in DB",
    )

    #
    # Optimization parameters
    #
    parser.add_argument(
        "--method",
        type=str,
        choices=["LP", "WF", "SA"],
        required=True,
        help="Choose between Linear Programming (LP) and Simulated Annealing (SA)",
    )
    parser.add_argument(
        "--sa-technique",
        type=str,
        choices=["Nullspace", "Kpaths"],
        help="For SA, choose between technique used",
    )
    parser.add_argument(
        "-s",
        "--solution",
        type=str,
        default="MinCost",
        choices=["Viable", "MinCost", "MinCostNorm", "Balancing", "QueueDelay"],
        help="Chooses the type of problem to be solved",
    )
    parser.add_argument(
        "--no-improvement-limit",
        type=int,
        default=-1,
        help="Maximum number of transitions where, if no improvements are made, the algorithms stop executing",
    )
    parser.add_argument(
        "--candidate-function",
        type=str,
        default="IncDec",
        help="Type of candidate creation function",
    )
    parser.add_argument(
        "--temperature-decay",
        type=str,
        choices=["log", "exp99"],
        default="log",
        help="Type of temperature decay function to use",
    )
    parser.add_argument(
        "--variable-initialization",
        type=str,
        choices=["random", "zeros", "sp"],
        default="random",
        help="Process to initialize the variables",
    )
    return parser.parse_args()


args = ParseArguments()
# Network used to optimize
graphName = args.graph_name
# Chosen solution
chosenSolution = args.solution
# SA arguments
K = args.K if type(args.K) is list else [args.K]
N = args.N if type(args.N) is list else [args.N]
T0 = args.T0 if type(args.T0) is list else [args.T0]
epsilon = args.epsilon if type(args.epsilon) is list else [args.epsilon]
sa_technique = args.sa_technique
numPaths = (
    args.number_of_paths
    if type(args.number_of_paths) is list
    else [args.number_of_paths]
)
method = args.method
noLpOptimal = args.no_lp_optimal
minimumGap = args.minimum_gap
progressBar = args.progress_bar
noSqlSave = args.no_sql_save
output = args.output
output_id = args.output_id
outputGraphs = args.output_graph
outputNetwork = args.output_network
saveStep = args.save_step
noImprovementLimit = args.no_improvement_limit
candidateFunction = args.candidate_function
variable_initialization = args.variable_initialization
temperatureDecay = args.temperature_decay

# Checks if its a valid graph before continuing
if graphName[-5:] != ".json":
    print("ERROR: Only .json files are valid network names")
    print(f"ERROR: Invalid '{graphName}'")
    exit(1)


# Creates a table to wich store the results
def GetGraphs(sa):
    graphs = {}
    for graphName in [
        "jHist",
        "jMinHist",
        "acceptanceHist",
        "acceptanceProbabilitySteps",
    ]:
        if hasattr(sa, graphName):
            graphs[graphName] = getattr(sa, graphName).copy()
    return graphs


#
# Execução do algoritmo e solução
#

for n, k, t0, ep, nPaths in product(N, K, T0, epsilon, numPaths):
    if method == "SA":
        a = SANetwork(graphName)

        preparationTime = -tm.process_time()

        if chosenSolution == "MinCost":
            objective = "MinCost"
            if sa_technique == "Nullspace":
                sol = MinCostSolution(a, x0=variable_initialization)
            elif sa_technique == "Kpaths":
                sol = KPathsMinimization(a, nPaths)
            else:
                raise NotImplementedError(
                    "SA technique not implemented for this solution"
                )
            # Gets the optimal value to early stop the simulated annealing algorithm if optimal is found
            optimalCost = FindOptimal(graphName)
            if not noLpOptimal and not optimalCost:
                lpSol = LPNetwork(graphName)
                lpSol.SolveLP()
                optimalCost = lpSol.GetLPOptimalCost()
            else:
                optimalCost = None

        elif chosenSolution == "MinCostNorm":
            objective = "MinCostNorm"
            if sa_technique == "Nullspace":
                sol = MinCostNormSolution(a, x0=variable_initialization)
            elif sa_technique == "Kpaths":
                sol = KPathsMinimizationNorm(a, nPaths)
            else:
                raise NotImplementedError(
                    "SA technique not implemented for this solution"
                )
            optimalCost = None

        elif chosenSolution == "QueueDelay":
            objective = "QueueDelay"
            if sa_technique == "Nullspace":
                sol = QueueDelayMinimization(a, x0=variable_initialization)
            elif sa_technique == "Kpaths":
                sol = KPathsQueueDelayMinimization(a, nPaths)
            else:
                raise NotImplementedError(
                    "SA technique not implemented for this solution"
                )
            optimalCost = None

        elif chosenSolution == "Balanced":
            objective = "Balancing"
            sol = BalancedSolution(a)
            optimalCost = None

        elif chosenSolution == "Viable":
            objective = "ViableSolution"
            sol = ViableSolution(a, x0="zero")
            optimalCost = 0

        else:
            raise NotImplementedError("SA was not implemented for the desired solution")

        sol.SwitchCandidate(candidateFunction)

        if temperatureDecay == "log":
            tempFunction = lambda t0, k: t0 / np.log(2 + k)
        elif temperatureDecay == "exp99":
            tempFunction = lambda t0, k: t0 * ((0.99) ** k)
        else:
            raise NotImplementedError(
                f"Temperature decay function '{temperatureDecay}' not yet implemented"
            )

        sa = SimulatedAnnealing(
            sol,
            n,
            k,
            t0,
            ep,
            gap=minimumGap,
            optimalValue=optimalCost,
            saveStep=saveStep,
            maxNoImprovements=noImprovementLimit,
            progressBar=progressBar,
            temperatureDecay=tempFunction,
        )

        preparationTime += tm.process_time()

        initialCost = sol.j
        initialAnswer = sol.x

        executionTime = -tm.time()
        sa.Execute()
        executionTime += tm.time()

        res = sa.solution.xmin

        costFunctionDelay = sa.solution.costDelay / sa.solution.costDelayMeasures
        candidateFunctionDelay = (
            sa.solution.candidateDelay / sa.solution.candidateDelayMeasures
        )

        foundCost = sa.solution.GetMinimumCost()
        numberVariables = np.prod(sa.solution.xmin.shape)

        timeToViableSolution = sa.GetTimeToViableSolution()
        averageAnswerImprovementTime = sa.GetAverageAnswerImprovementTime()
        viableSolutionFound = sa.solution.viable

    elif method == "WF":
        prepTimeStart = tm.perf_counter()
        a = SANetwork(graphName)
        if chosenSolution == "MinCost":
            sol = KPathsMinimization(a, nPaths)
        elif chosenSolution == "QueueDelay":
            sol = KPathsQueueDelayMinimization(a, nPaths)
        else:
            raise NotImplementedError("WF Method not implemented for this objective")
        objective = chosenSolution
        preparationTime = tm.perf_counter() - prepTimeStart

        startTime = tm.perf_counter()
        foundCost, viableSolutionFound = GetSolutionWaterFilling(sol)
        executionTime = tm.perf_counter() - startTime
        sa_technique = None
        optimalCost = None
        timeToViableSolution = executionTime if viableSolutionFound else None
        averageAnswerImprovementTime = executionTime if viableSolutionFound else None

    elif method == "LP":
        a = LPNetwork(graphName)
        initialCost = -1
        numberVariables = len(a.GetLinksList()) * len(a.ListContracts())
        if chosenSolution == "MinCost":
            objective = "MinCost"
            a = LPNetwork(graphName)
            a.SolveLP()
            foundCost = a.GetLPOptimalCost()
            optimalCost = foundCost
            preparationTime = a.GetLPPreparationTime()
            executionTime = a.GetLastLPExecutionTime()
        else:
            raise Exception("Invalid Linear Programming solution")

        timeToViableSolution = executionTime
        averageAnswerImprovementTime = executionTime
        viableSolutionFound = True

    else:
        raise Exception("Invalid method")

    #
    # Adding results to table
    #
    results = pd.Series()
    results["name"] = Path(graphName).name
    results["method"] = method.upper()
    results["sa_technique"] = sa_technique
    results["objective"] = objective
    results["k"] = k if method == "SA" else None
    results["n"] = n if method == "SA" else None
    results["t0"] = t0 if method == "SA" else None
    results["epsilon"] = ep if method == "SA" else None
    results["number_of_paths"] = nPaths if sa_technique == "Kpaths" else None
    results["optimal_cost"] = optimalCost
    results["found_cost"] = foundCost
    results["gap_to_optimal"] = ((foundCost / optimalCost) - 1) if optimalCost else None
    results["preparation_time"] = preparationTime
    results["execution_time"] = executionTime
    results["viable_solution_found"] = viableSolutionFound
    results["time_to_viable_solution"] = timeToViableSolution
    results["average_answer_improvement_time"] = averageAnswerImprovementTime
    results["maquina"] = socket.gethostname()
    results["candidate_function"] = candidateFunction
    results["variable_initialization"] = variable_initialization
    results["temperature_decay_function"] = temperatureDecay

    results = results.dropna()

    resultRowId = None
    if not noSqlSave:
        resultRowId = save_data(results, GetGraphs(sa) if method == "SA" else None)

    if output:
        print(results)

    if output_id:
        if not resultRowId:
            print("ERROR: No result id returned")
        print(resultRowId)

    if outputGraphs:
        if method == "LP":
            print("WARNING: Linear Programming doesn't have metric graphs")
        elif method == "SA":
            plt.plot(sa.jHist[10:-5])
            plt.title(r"Variação de $J(\mathbf{x})$ ao longo da execução")
            plt.ylabel(r"$J(\mathbf{x})$")
            plt.xlabel("Passos")
            # plt.yscale("log")
            plt.show()

            sa.PlotAcceptanceRateSteps()
            plt.title("Acceptance rate between steps")
            plt.show()

            sa.PlotJMinTransitions()
            plt.title("J min evolution between transitions")
            plt.yscale("log")
            plt.show()

            sa.PlotAcceptanceRateTransitions()
            plt.title("Acceptance rate between transitions")
            plt.show()
