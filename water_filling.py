import sys
import os

import numpy as np
import pandas as pd

from network import SANetwork
from save_data_sql import FindOptimal
from simulated_annealing.graph_solutions import (
    KPathsMinimization,
    KPathsQueueDelayMinimization,
)


def GetSolutionWaterFilling(sol, attempts=10):
    """
    This heuristic tries to find the best solution by incrementing the demand on each contract until it
    finds a viable solution or it fails to find a feasible solution.
    """
    capacityVector = sol.capacityVector
    demandVector = sol.demandVector

    metDemands = False

    x = np.zeros(sol.x.shape)
    for i in range(attempts):
        print(f"Attempt #{i}")
        x = np.zeros(sol.x.shape)

        while True:
            acceptedModification = False

            for contract in np.random.choice(
                range(x.shape[0]), x.shape[0], replace=False
            ):
                # Starts with the first path
                validPaths = np.ones(i + 1) * (-1)
                currentCost = np.ones(i + 1) * np.inf

                # If there's demand to be allocated to this contract
                if np.sum(x, axis=1)[contract, 0] < demandVector[contract]:
                    # Evaluates the cost function for every path,
                    # chooses path wich minimizes change in the cost
                    # function
                    for path in range(x.shape[1]):
                        x[contract, path, 0] += 1
                        # If path breaks constraints, ignore path
                        if sol._CapacityReachedPenalty(x) > 1e-10:
                            x[contract, path, 0] -= 1
                            continue
                        # Keeps the best n paths found
                        newCost = sol._Cost(x)
                        for idx, cost in enumerate(currentCost):
                            if newCost < cost:
                                currentCost[idx] = newCost
                                validPaths[idx] = path
                                break

                        # Reverts to original answer
                        x[contract, path, 0] -= 1

                # After verifying every path, randomly selects between the best three paths
                if (validPaths > -1).any():
                    probPaths = currentCost[currentCost < np.inf]
                    probPaths = np.max(probPaths) - probPaths + 1
                    probPaths = probPaths / np.sum(probPaths)
                    chosenPath = np.random.choice(
                        validPaths[validPaths > -1], p=probPaths
                    )
                    x[contract, int(chosenPath), 0] += 1
                    acceptedModification = True

            if not acceptedModification:
                break

            if (np.sum(x, axis=1)[:, 0] == demandVector).all():
                metDemands = True
                break

        if metDemands:
            break

    sol.xmin = x
    return sol.GetMinimumCost(), metDemands


if __name__ == "__main__":
    results = []

    graphPaths = sys.argv[1]
    maxPaths = 30

    for i in os.listdir(graphPaths):
        a = SANetwork(graphPaths + i)
        sol = KPathsQueueDelayMinimization(a, maxPaths)
        cost, metDemands = GetSolutionWaterFilling(sol)
        print(i, cost, metDemands)
        lpCost = FindOptimal(i)
        results.append(
            {
                "graph": i,
                "metDemands": metDemands,
                "cost_wf": cost,
                "cost_lp": lpCost,
                "gap": (cost / lpCost) - 1,
            }
        )

    results = pd.DataFrame(results)
    print(results)
