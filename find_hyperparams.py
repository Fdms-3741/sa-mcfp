#!/usr/bin/env python3

# find_hyperparams.py
#
# This script finds the best hyperparameters
import sys
import subprocess
from pathlib import Path
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from save_data_sql import GetOptimalValue

COMPUTE_STEPS = 1000000

bounds = {"temp": (0, 3), "k": (1, 3), "epsilon": (1, 100.0)}


def GetParameters(temp, k):
    temp = int(np.ceil(10**temp))
    k = int(np.ceil(10**k))
    n = int(np.round(COMPUTE_STEPS // k, -1))
    return temp, k, n


def SolveSAProblem(temp: float, k: float, epsilon: float) -> float:
    temp, k, n = GetParameters(temp, k)

    proc = subprocess.run(
        [
            "python3.13",
            "sa_solver.py",
            "--method",
            "SA",
            "-s",
            sys.argv[2],
            "--sa-technique",
            sys.argv[3],
            "--variable-initialization",
            "random",
            "--candidate-function",
            "IncDec",
            "--temperature-decay",
            "log",
            "--no-improvement-limit",
            f"{int(np.ceil(0.4*k))}",
            "--save-step",
            "100",
            "-T",
            str(temp),
            "-N",
            str(n),
            "-K",
            str(k),
            "--epsilon",
            str(epsilon),
            "--no-lp-optimal",
            "--output-id",
            sys.argv[1],
        ],
        capture_output=True,
    )

    if proc.returncode:
        print("EXECUTION ERROR")
        print(proc.stderr)
        return 1e-10

    res = GetOptimalValue(int(proc.stdout))

    return -res if res else 1e-10


if __name__ == "__main__":
    opt = BayesianOptimization(
        f=SolveSAProblem,
        pbounds=bounds,
    )

    graphName = Path(sys.argv[1]).name
    fileName = f"./results/hyperparameter_search/{graphName}-{sys.argv[2]}-{sys.argv[3]}-bayesopt.log"

    if Path(fileName).exists():
        load_logs(opt, [fileName])

    # Saves result
    logger = JSONLogger(path=fileName)
    opt.subscribe(Events.OPTIMIZATION_STEP, logger)

    opt.maximize(n_iter=10, init_points=3)

    print("Results: ", opt.max)
    if opt.max:
        temp, k, n = GetParameters(opt.max["params"]["temp"], opt.max["params"]["k"])
        print(
            f"Parameters used: k={k}, n={n}, temp={temp}, epsilon={opt.max['params']['epsilon']}"
        )
