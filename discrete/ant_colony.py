import numpy as np
import random
from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution
from jmetal.core.algorithm import Algorithm
from jmetal.util.observable import DefaultObservable
from typing import List

from matplotlib import pyplot as plt

from discrete.tsp_problem import TSPProblem

from jmetal.core.algorithm import Algorithm
from jmetal.core.solution import PermutationSolution
from jmetal.util.observable import DefaultObservable
import numpy as np
import random


class AntColonyOptimization(Algorithm):
    def __init__(self, problem, n_ants=10, n_iterations=100,
                 alpha=1.0, beta=2.0, evaporation_rate=0.5, q=100.0):

        super().__init__()
        self.problem = problem
        self.n_ants = n_ants
        self.max_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.q = q

        self.pheromone = np.ones((self.problem.number_of_variables(), self.problem.number_of_variables()))
        self.distance_matrix = self.compute_distance_matrix()
        self.iteration = 0
        self._best_solution = None
        self.best_objectives = []
        self.observable = DefaultObservable()

    def compute_distance_matrix(self):
        coords = self.problem.coordinates
        n = len(coords)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        return matrix

    def create_initial_solutions(self):
        solutions = []
        for _ in range(self.n_ants):
            permutation = list(range(self.problem.number_of_variables()))
            np.random.shuffle(permutation)
            solution = PermutationSolution(
                number_of_variables=self.problem.number_of_variables(),
                number_of_objectives=self.problem.number_of_objectives()
            )
            solution.variables = permutation
            solutions.append(solution)
        return solutions

    def evaluate(self, solutions):
        return [self.problem.evaluate(sol) for sol in solutions]

    def init_progress(self):
        self.iteration = 0
        self._best_solution = None
        self.best_objectives = []

    def update_progress(self):
        self.iteration += 1
        self.best_objectives.append(self._best_solution.objectives[0])
        self.observable.notify_all(self)

    def stopping_condition_is_met(self):
        return self.iteration >= self.max_iterations

    def result(self):
        return self._best_solution

    def get_name(self):
        return "ClassicACO"

    def observable_data(self):
        return self.observable

    def step(self):
        solutions = []

        for _ in range(self.n_ants):
            visited = [False] * self.problem.number_of_variables()
            current = np.random.randint(0, self.problem.number_of_variables() - 1)
            tour = [current]
            visited[current] = True

            for _ in range(self.problem.number_of_variables() - 1):
                probabilities = []
                for j in range(self.problem.number_of_variables()):
                    if not visited[j]:
                        tau = self.pheromone[current][j] ** self.alpha
                        eta = (1 / self.distance_matrix[current][j]) ** self.beta
                        probabilities.append((j, tau * eta))

                total = sum(prob for _, prob in probabilities)
                if total == 0:
                    next_city = np.random.choice(
                        [j for j in range(self.problem.number_of_variables()) if not visited[j]])
                else:
                    r = np.random.uniform(0, total)
                    cumulative = 0
                    for j, prob in probabilities:
                        cumulative += prob
                        if r <= cumulative:
                            next_city = j
                            break

                tour.append(next_city)
                visited[next_city] = True
                current = next_city

            solution = PermutationSolution(
                number_of_variables=self.problem.number_of_variables(),
                number_of_objectives=self.problem.number_of_objectives()
            )
            solution.variables = tour
            self.problem.evaluate(solution)
            solutions.append(solution)

            if self._best_solution is None or solution.objectives[0] < self._best_solution.objectives[0]:
                self._best_solution = solution

        self.evaporate_pheromone()
        self.deposit_pheromone(solutions)

    def evaporate_pheromone(self):
        self.pheromone *= (1.0 - self.rho)

    def deposit_pheromone(self, solutions):
        for sol in solutions:
            dist = sol.objectives[0]
            for i in range(len(sol.variables)):
                from_city = sol.variables[i]
                to_city = sol.variables[(i + 1) % len(sol.variables)]
                self.pheromone[from_city][to_city] += self.q / dist
                self.pheromone[to_city][from_city] += self.q / dist  # for symmetry


if __name__ == '__main__':
    np.random.seed(42)
    coords = np.random.rand(20, 2)
    problem = TSPProblem(coords)
    algorithm = AntColonyOptimization(
        problem=problem,
        n_ants=10,
        n_iterations=100,
    )
    algorithm.init_progress()

    while not algorithm.stopping_condition_is_met():
        algorithm.step()
        algorithm.update_progress()
        # if algorithm.iteration % 10 == 0:  # Print every 10 iterations
        #     print(
        #         f"Iteration {algorithm.iteration}, Best: {algorithm._best_solution.objectives[0] if algorithm._best_solution else 'None'}")

    result = algorithm.result()
    print("Najlepsza trasa:", result.variables)
    print("Długość trasy:", result.objectives[0])
    problem.plot_solution(result)

# TypeError: Can't instantiate abstract class AntColonyOptimization without an implementation for abstract methods 'create_initial_solutions', 'evaluate', 'get_name', 'init_progress', 'observable_data', 'result', 'step', 'stopping_condition_is_met', 'update_progress'
