# sa_tsp.py
# Aluno: Fernando Dias
#
# Esse código tem a implementação do simulated annealing para resolver o problema do caixeiro viajante
#
#
from abc import ABC, abstractmethod

import time as tm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Solution(ABC):
    """
    This class administers the solutions for the Simulated Annealing algorithm. This class is responsible for:

    * Storing the current, minimal and candidate solutions and their respective energy values.
    * Implementing the cost function to evaluate a solution
    * Implementing the candidate calculation function
    * (Optional) Implement a function to verify answer's feasibility and correct the current value if necessary
    """

    @abstractmethod
    def __init__(self, x):
        pass

    @property
    def x(self):
        """Current solution. Used in the SA algorithm."""
        return self._x

    @x.setter
    def x(self, x):
        self._x = x.copy()
        self._j = self._Cost(x)

    @property
    def xmin(self):
        """Best answer found so far"""
        return self._xmin

    @xmin.setter
    def xmin(self, xmin):
        self._xmin = xmin.copy()
        self._jmin = self._Cost(xmin)
        self._viable = self._IsViable(xmin)

    @property
    def xhat(self):
        """Candidate answer"""
        return self._xhat

    @xhat.setter
    def xhat(self, xhat):
        self._xhat = xhat.copy()
        self._jhat = self._Cost(xhat)

    @property
    def j(self):
        """Cost of the current answer"""
        return self._j

    @property
    def jmin(self):
        """Cost of the best answer"""
        return self._jmin

    @property
    def jhat(self):
        """Cost of the candidate answer"""
        return self._jhat

    @abstractmethod
    def Candidate(self, epsilon):
        """Based on the current answer ``x``, calculates the candidate ``xhat``."""
        pass

    def Accept(self):
        """Accepts the current candidate as new current answer. If cost is less than the minimum cost, candidate also becomes the best ``xmin`` answer."""
        self._x = self._xhat
        self._j = self._jhat
        if self.j < self.jmin:
            self.xmin = self.x
            return True
        return False

    def Initialize(self):
        self.xhat = self.x
        self.xmin = self.x

    @property
    def viable(self):
        """Returns whether the current minimal solution found is a viable solution"""
        return self._viable

    @abstractmethod
    def _IsViable(self, x) -> bool:
        """Function that evaluates whether xmin is viable"""
        pass

    @abstractmethod
    def _Cost(self, x) -> float:
        """Calculates the cost of the answer x"""
        pass


class SimulatedAnnealing:
    """
    This class implements the simulated annealing algorithm, given a Solution class and the parameters.

    To start the algorithm, parameters, K, N, T0 and epsilon must be defined.

    Optionally, the optimal value (if known) and the gap to optimal value can be defined, and they will be used as stop conditions.

    A stop condition can be defined in the solution class, for this, the class must define a `stopAlgorithm` boolean variable and a `stopCondition` string with the stop condition description.
    """

    def __init__(
        self,
        solution,
        N,
        K,
        T0,
        epsilon,
        temperatureDecay=None,
        saveStep=None,
        maxNoImprovements=None,
        optimalValue=None,
        gap=None,
        maxTime=None,
        progressBar=False,
    ):
        self.solution = solution
        self.K = K
        self.N = N
        self.T0 = T0
        self.epsilon = epsilon
        self.step = saveStep
        # Termos para saída antecipada
        self.optimalValue = optimalValue
        self.desiredGap = gap
        # Maximum number of transitions where no improvement is made
        self.startStallConditionVerification = int(np.ceil(self.K / 3))
        self.maximumNoImprovementTransitions = maxNoImprovements
        # Variáveis para observação de congelamento
        self.statesFrozenCount = 0
        self.lastChangeCount = 0
        # Variáveis para transição
        self.jTransitions = np.zeros((K,))
        self.jminTransitions = np.zeros((K,))
        self.acceptanceTransitions = np.zeros((K,))
        self.xTransitions = [None] * K
        self.maximumExecutionTime = maxTime
        self.progressBar = progressBar
        self.betterMinimumSolutionsCount = 0  # Counts the number of times that the accepted solution is the new minimum

        self.acceptedChangesOnTemperature = 0
        if temperatureDecay:
            self.TemperatureDecay = temperatureDecay
        else:
            self.TemperatureDecay = lambda T0, k: T0 / np.log2(2 + k)

    @property
    def step(self):
        """The step property."""
        return self._step

    @step.setter
    def step(self, value):
        self._step = value
        if value is not None:
            historySize = self.K * self.N // value
            self.jHist = np.zeros(historySize + 5)
            self.jMinHist = np.zeros(historySize + 5)
            self.acceptanceHist = np.zeros(historySize + 5)
            self.acceptanceProbabilitySteps = np.zeros(historySize + 5)

    def SaveStateAtStep(self, k, n):
        if self._step and (n % self._step) == 0:
            self.jHist[(k * self.N + n) // self._step] = self.solution.j
            self.jMinHist[(k * self.N + n) // self._step] = self.solution.jmin
            self.acceptanceHist[(k * self.N + n) // self._step] = (
                self.acceptedChangesOnTemperature / (n + 1)
            )
            self.acceptanceProbabilitySteps[(k * self.N + n) // self._step] = (
                self.acceptanceProbability
            )

    def CalculateAcceptanceProbability(self, T):
        self.acceptanceProbability = np.min(
            [np.exp((self.solution.j - self.solution.jhat) / T), 1.001]
        )

    def SaveCurrentTransition(self, k):
        self.jTransitions[k] = self.solution.j
        self.jminTransitions[k] = self.solution.jmin
        self.acceptanceTransitions[k] = self.acceptedChangesOnTemperature / self.N
        self.xTransitions[k] = self.solution.x.copy()

    def SuccessConditionsReached(self):
        if self.OptimumValueReached():
            self._stopCondition = "Optimal answer found"
            return True

        if self.GapReached():
            self._stopCondition = "Gap reached"
            return True

        if self.SolutionStopConditionReached():
            self._stopCondition = self.solution.stopCondition
            return True

        if self.MaxTimeReached():
            self._stopCondition = "Max time reached"

        return False

    def SolutionStopConditionReached(self):
        return self.solution.stopAlgorithm

    def OptimumValueReached(self):
        return (self.optimalValue is not None) and np.abs(
            self.optimalValue - self.solution.jmin
        ) < 1e-10

    def GapReached(self):
        return (
            (self.desiredGap is not None)
            and (self.optimalValue is not None)
            and np.abs(self.optimalValue / self.solution.jmin) > self.desiredGap
        )

    def MaxTimeReached(self):
        if (
            self.maximumExecutionTime
            and tm.time() - self.start_time > self.maximumExecutionTime
        ):
            return True
        return False

    def FrozenStateReached(self):
        raise NotImplementedError

    def MaxNoImprovementTransitions(self):
        """
        If maximumNumberOfTransition is set and its reached, stops the algorithm
        """
        return (
            self.maximumNoImprovementTransitions
            and self.maximumNoImprovementTransitions > 0
            and (self._k - self.lastImprovementTransition)
            > self.maximumNoImprovementTransitions
        )

    def StallConditionsReached(self):
        """
        Implements early detection of frozen state
        """
        if self._k > self.startStallConditionVerification:
            if self.MaxNoImprovementTransitions():
                self._stopCondition = "Maximum no improvement transitions reached"
                return True
            return False

    def PlotJSteps(self):
        fig, ax = plt.subplots()
        ax.plot(self.jHist[:-5])
        ax.set_title("J Value")
        return fig, ax

    def PlotAcceptanceRateSteps(self):
        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(self.acceptanceHist[:-5])), self.acceptanceHist[:-5])
        ax.set_title("Acceptance rate")
        return fig, ax

    def PlotJTransitions(self):
        fig, ax = plt.subplots()
        ax.plot(self.jTransitions[:-5])
        ax.set_title("J on transitions")
        return fig, ax

    def PlotJMinTransitions(self):
        fig, ax = plt.subplots()
        ax.plot(self.jminTransitions[:-5])
        ax.set_title(r"$J_{\min}$ on transitions")
        return fig, ax

    def PlotAcceptanceRateTransitions(self):
        fig, ax = plt.subplots()
        ax.scatter(
            np.arange(len(self.acceptanceTransitions[:-5])),
            self.acceptanceTransitions[:-5],
        )
        ax.set_title("Acceptance rate on transitions")
        return fig, ax

    def PlotAcceptanceProbabilitySteps(self):
        fig, ax = plt.subplots()
        ax.scatter(
            np.arange(len(self.acceptanceProbabilitySteps[:-5])),
            self.acceptanceProbabilitySteps[:-5],
        )
        ax.set_title("Acceptance probability steps")
        return fig, ax

    @property
    def stopCondition(self):
        return self._stopCondition

    def GetExecutedSteps(self):
        """
        When an early stop is reached, this function returns the current k-th and n-th step of the algorithm and the temperature.
        """
        return self._k, self._n, self._t

    def EstimateTotalSteps(self):
        return self.K * self.N

    def Execute(self):
        self.start_time = tm.time()
        self.startViableSolutionTime = tm.time()
        self.lastBetterSolutionFoundTime = np.inf
        self.lastImprovementTransition = 0
        self.timeToViableSolution = 0
        self._stopCondition = None
        if self.progressBar:
            pbar = tqdm(total=self.EstimateTotalSteps())
        stopAlgorithm = False

        # If problem starts with a viable solution, it doesn't measure the time to viable time
        self.viableSolutionFound = self.solution.viable

        for self._k in range(self.K):
            self.SaveCurrentTransition(self._k)
            self._t = self.TemperatureDecay(self.T0, self._k)
            self.acceptedChangesOnTemperature = 0
            for self._n in range(self.N):
                if self.progressBar:
                    pbar.update(1)
                self.solution.Candidate(self.epsilon)
                self.CalculateAcceptanceProbability(self._t)
                if np.random.uniform(0, 1) < self.acceptanceProbability:
                    self.acceptedChangesOnTemperature += 1
                    # If xmin was modified, accept returns true
                    if self.solution.Accept():
                        self.lastImprovementTransition = self._k
                        # To find the average best solution found time
                        if self.viableSolutionFound:
                            self.betterMinimumSolutionsCount += 1
                            self.lastBetterSolutionFoundTime = tm.time()
                        # To find the time to viable solution
                        if not self.viableSolutionFound and self.solution.viable:
                            self.timeToViableSolution = tm.time() - self.start_time
                            self.viableSolutionFound = True
                            self.startViableSolutionTime = tm.time()

                    # Verifies early stop conditions
                    if self.SuccessConditionsReached():
                        self.duration = tm.time() - self.start_time
                        stopAlgorithm = True
                        break
                else:
                    if self.StallConditionsReached():
                        self.duration = tm.time() - self.start_time
                        stopAlgorithm = True
                        break

                self.SaveStateAtStep(self._k, self._n)
            # Stops algorithm
            if stopAlgorithm:
                break

        self.duration = tm.time() - self.start_time
        self.averageAnswerImprovementTime = (
            (self.lastBetterSolutionFoundTime - self.startViableSolutionTime)
            / self.betterMinimumSolutionsCount
            if self.betterMinimumSolutionsCount > 0
            else np.inf
        )
        if not self.stopCondition:
            self._stopCondition = "Fim da execução"

    def GetAverageAnswerImprovementTime(self):
        return self.averageAnswerImprovementTime

    def GetTimeToViableSolution(self):
        return self.timeToViableSolution


if __name__ == "__main__":

    class Rastrigin(Solution):
        def __init__(self, dimensions):
            self.n = dimensions
            self._x = np.random.uniform(-5, 5, size=(dimensions,))
            self._xmin = self.x
            self._xhat = self.x
            self._j = self._Cost(self._x)
            self._jmin = self.j
            self._jhat = self.j
            self.stopAlgorithm = False
            self._viable = True

        def _Cost(self, x):
            A = 10
            return A * self.n + np.sum(np.power(x, 2) - A * np.cos(2 * np.pi * x))

        def _IsViable(self, x):
            return True

        def Candidate(self, epsilon):
            self._xhat = self._x + epsilon * np.random.normal(0, 1, size=(self.n,))
            self._jhat = self._Cost(self._xhat)

    N = 50000
    K = 10
    T0 = 20
    epsilon = 0.5

    sa = SimulatedAnnealing(
        Rastrigin(5),
        N,
        K,
        T0,
        epsilon,
        saveStep=10,
        optimalValue=0,
        gap=0.9,
        # temperatureDecay=lambda t0, k: t0 * (0.99) ** k,
    )

    print(f"Estado atual para J={sa.solution.jmin}")
    print(sa.solution.xmin)
    print("Executando o SA")
    sa.Execute()
    print(f"Solução para J={sa.solution.jmin}")
    print(sa.solution.xmin)
    print(f"TTV: {sa.timeToViableSolution}")
    print(f"Average minimization time: {sa.averageAnswerImprovementTime}")

    fig, ax = sa.PlotJSteps()
    plt.show()
    fig, ax = sa.PlotAcceptanceRateSteps()
    plt.show()

    fig, ax = sa.PlotJTransitions()
    plt.show()
    fig, ax = sa.PlotAcceptanceRateTransitions()
    plt.show()

    fig, ax = sa.PlotAcceptanceProbabilitySteps()
    plt.show()
