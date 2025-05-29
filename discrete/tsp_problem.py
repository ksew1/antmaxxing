from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution
import numpy as np
from matplotlib import pyplot as plt
import random


class TSPProblem(PermutationProblem):
    def __init__(self, coordinates):
        super().__init__()
        self.coordinates = coordinates

    def number_of_variables(self) -> int:
        return len(self.coordinates)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def name(self) -> str:
        return "TSP"

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        dist = 0.0
        permutation = solution.variables
        for i in range(len(permutation)):
            a = self.coordinates[permutation[i]]
            b = self.coordinates[permutation[(i + 1) % len(permutation)]]
            dist += np.linalg.norm(a - b)
        solution.objectives[0] = dist
        return solution

    def create_solution(self) -> PermutationSolution:
        permutation = list(range(self.number_of_variables()))
        random.shuffle(permutation)
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables(),
                                           number_of_objectives=self.number_of_objectives())
        new_solution.variables = permutation
        return new_solution

    def plot_solution(self, solution):
        """Plot the TSP tour"""
        plt.figure(figsize=(10, 6))

        # Plot cities
        x_coords = [self.coordinates[i][0] for i in range(self.number_of_variables())]
        y_coords = [self.coordinates[i][1] for i in range(self.number_of_variables())]
        plt.scatter(x_coords, y_coords, color='blue', s=100)

        # Plot tour
        for i in range(self.number_of_variables()):
            city_from = solution.variables[i]
            city_to = solution.variables[(i + 1) % self.number_of_variables()]

            plt.plot([self.coordinates[city_from][0], self.coordinates[city_to][0]],
                     [self.coordinates[city_from][1], self.coordinates[city_to][1]],
                     'k-', alpha=0.5)

        plt.title(f'TSP Tour (Total Distance: {solution.objectives[0]:.2f})')
        plt.grid()
        plt.show()
