from matplotlib import pyplot as plt

import numpy as np
import random
from jmetal.core.solution import PermutationSolution
from jmetal.core.algorithm import Algorithm
from jmetal.util.observable import DefaultObservable
from typing import List

from discrete.tsp_problem import TSPProblem


class DualPheromoneACO(Algorithm):
    def get_name(self) -> str:
        return "Dual Pheromone ACO"

    def __init__(self,
                 problem: TSPProblem,
                 n_ants=10,
                 n_iterations=100,
                 alpha=1.0,
                 beta=2.0,
                 gamma=2.0,
                 rho_pos=0.1,
                 rho_neg=0.1,
                 n_reinforce=2,
                 elite_boost=False,
                 exploration_rate=0.1):
        super().__init__()
        self.problem = problem
        self.n_ants = n_ants
        self.max_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho_pos = rho_pos
        self.rho_neg = rho_neg
        self.n_reinforce = n_reinforce
        self.elite_boost = elite_boost
        self.exploration_rate = exploration_rate

        self.n = problem.number_of_variables()
        self.tau_pos = np.ones((self.n, self.n))
        self.tau_neg = np.zeros((self.n, self.n))
        self.distance_matrix = self.compute_distance_matrix()

        self.observable = DefaultObservable()
        self.iteration = 0
        self.evaluations = 0
        self._best_solution = None
        self.convergence_history = []

    def compute_distance_matrix(self):
        coords = self.problem.coordinates
        n = len(coords)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        return matrix

    def create_initial_solutions(self) -> List[PermutationSolution]:
        return [self.construct_solution() for _ in range(self.n_ants)]

    def evaluate(self, solution_list: List[PermutationSolution]) -> List[PermutationSolution]:
        return [self.problem.evaluate(solution) for solution in solution_list]

    def init_progress(self) -> None:
        self.iteration = 0
        self.evaluations = 0
        self.convergence_history = []
        self.observable.notify_all(self.observable_data())

    def stopping_condition_is_met(self) -> bool:
        return self.iteration >= self.max_iterations

    def step(self) -> None:
        solutions = [self.construct_solution() for _ in range(self.n_ants)]
        evaluated = self.evaluate(solutions)
        self.evaluations += len(evaluated)

        for s in evaluated:
            if self._best_solution is None or s.objectives[0] < self._best_solution.objectives[0]:
                self._best_solution = s

        self.update_pheromones(evaluated)
        self.iteration += 1

        if self._best_solution:
            self.convergence_history.append(self._best_solution.objectives[0])

    def update_progress(self) -> None:
        self.observable.notify_all(self.observable_data())

    def observable_data(self) -> dict:
        return {
            "ITERATION": self.iteration,
            "EVALUATIONS": self.evaluations,
            "COMPUTING_TIME": 0.0,
            "SOLUTIONS": [self._best_solution] if self._best_solution else []
        }

    def result(self) -> PermutationSolution:
        return self._best_solution

    def construct_solution(self) -> PermutationSolution:
        unvisited = list(range(self.n))
        current = np.random.choice(unvisited, size=1)[0]
        tour = [current]
        unvisited.remove(int(current))

        while unvisited:
            if np.random.random() < self.exploration_rate:
                # Eksploracja: wybór preferujący słaby negatywny feromon
                neg_values = np.array([
                    self.tau_neg[current][j] for j in unvisited
                ])
                # Zamieniamy na "atrakcyjność": im mniejszy negatywny, tym lepiej
                attractiveness = 1.0 / (1e-6 + neg_values)  # +epsilon by uniknąć dzielenia przez zero
                attractiveness /= attractiveness.sum()
                next_city = np.random.choice(unvisited, p=attractiveness)
            else:
                # Eksploatacja: klasyczny wybór ze wzorem z tau_pos, eta, tau_neg
                probs = []
                for j in unvisited:
                    tau = self.tau_pos[current][j] ** self.alpha
                    # eta = (1 / np.linalg.norm(
                    #     self.problem.coordinates[current] - self.problem.coordinates[j])) ** self.beta
                    eta = (1 / self.distance_matrix[current, j]) ** self.beta
                    psi = (1 + self.tau_neg[current][j]) ** self.gamma
                    probs.append((tau * eta) / psi)
                probs = np.array(probs)
                probs /= probs.sum()
                next_city = np.random.choice(unvisited, p=probs)

            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        solution = PermutationSolution(number_of_variables=self.n, number_of_objectives=1)
        solution.variables = tour
        return solution

    def update_pheromones(self, solutions: List[PermutationSolution]):
        self.tau_pos *= (1 - self.rho_pos)
        self.tau_neg *= (1 - self.rho_neg)

        solutions.sort(key=lambda s: s.objectives[0])
        good = solutions[:self.n_reinforce]
        bad = solutions[-self.n_reinforce:]

        # Zbierz krawędzie z najlepszych rozwiązań
        good_edges = set()
        for s in good:
            for i in range(self.n):
                a, b = s.variables[i], s.variables[(i + 1) % self.n]
                # Dodaj krawędź w obu kierunkach (symetryczna)
                good_edges.add((min(a, b), max(a, b)))

        # Standard reinforcement for good solutions
        for s in good:
            for i in range(self.n):
                a, b = s.variables[i], s.variables[(i + 1) % self.n]
                delta = 100 / s.objectives[0]
                self.tau_pos[a][b] += delta
                self.tau_pos[b][a] += delta

        # Elite boost for the best solution if enabled
        if self.elite_boost and good:
            best_solution = good[0]  # Already sorted, so first is best
            for i in range(self.n):
                a, b = best_solution.variables[i], best_solution.variables[(i + 1) % self.n]
                elite_delta = 100 / best_solution.objectives[0]
                self.tau_pos[a][b] += elite_delta
                self.tau_pos[b][a] += elite_delta

        # Negative reinforcement - tylko dla krawędzi NIE występujących w dobrych rozwiązaniach
        # Zliczamy wystąpienia każdej krawędzi w złych rozwiązaniach
        bad_edge_count = {}
        bad_edge_quality = {}

        for s in bad:
            solution_quality = 100 / s.objectives[0]  # Im gorsza trasa, tym mniejsza wartość
            for i in range(self.n):
                a, b = s.variables[i], s.variables[(i + 1) % self.n]
                edge = (min(a, b), max(a, b))

                # Tylko jeśli krawędź NIE występuje w dobrych rozwiązaniach
                if edge not in good_edges:
                    if edge not in bad_edge_count:
                        bad_edge_count[edge] = 0
                        bad_edge_quality[edge] = 0
                    bad_edge_count[edge] += 1
                    bad_edge_quality[edge] += solution_quality

        # Aktualizuj negatywne feromony proporcjonalnie do częstości i jakości
        max_count = max(bad_edge_count.values()) if bad_edge_count else 1

        for edge, count in bad_edge_count.items():
            a, b = edge
            avg_quality = bad_edge_quality[edge] / count
            frequency_factor = count / max_count  # 0-1 normalizacja częstości

            # Delta proporcjonalna do częstości wystąpień i odwrotnie do jakości
            delta = frequency_factor * (1 / avg_quality) * 0.1  # 0.1 jako scaling factor

            self.tau_neg[a][b] += delta
            self.tau_neg[b][a] += delta

    def plot_convergence(self):
        """Wykres zbieżności algorytmu"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.convergence_history) + 1), self.convergence_history, 'b-', linewidth=2)
        plt.title('Zbieżność algorytmu Dual Pheromone ACO')
        plt.xlabel('Iteracja')
        plt.ylabel('Długość najlepszej trasy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_pheromone_matrices(self, top_n=10):
        """Scatter plot: siła feromonu vs odległość między miastami"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Zbierz dane dla scatter plot
        distances = []
        pos_pheromones = []
        neg_pheromones = []

        for i in range(self.n):
            for j in range(i + 1, self.n):  # Tylko górny trójkąt (symetryczna macierz)
                dist = np.linalg.norm(self.problem.coordinates[i] - self.problem.coordinates[j])
                distances.append(dist)
                pos_pheromones.append(self.tau_pos[i][j])
                neg_pheromones.append(self.tau_neg[i][j])

        # Scatter plot dla pozytywnych feromonów
        ax1.scatter(distances, pos_pheromones, alpha=0.6, color='green', s=30)
        ax1.set_xlabel('Odległość między miastami')
        ax1.set_ylabel('Siła feromonu pozytywnego')
        ax1.set_title('Feromony pozytywne vs Odległość')
        ax1.grid(True, alpha=0.3)

        # Scatter plot dla negatywnych feromonów
        ax2.scatter(distances, neg_pheromones, alpha=0.6, color='red', s=30)
        ax2.set_xlabel('Odległość między miastami')
        ax2.set_ylabel('Siła feromonu negatywnego')
        ax2.set_title('Feromony negatywne vs Odległość')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_pheromone_city_map(self, top_positive=8, top_negative=8):
        """Mapa miast z najsilniejszymi feromonami jako linie"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Znajdź najsilniejsze pozytywne feromony
        pos_connections = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pos_connections.append((self.tau_pos[i][j], i, j))
        pos_connections.sort(reverse=True)
        top_pos = pos_connections[:top_positive]

        # Znajdź najsilniejsze negatywne feromony
        neg_connections = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                neg_connections.append((self.tau_neg[i][j], i, j))
        neg_connections.sort(reverse=True)
        top_neg = neg_connections[:top_negative]

        # Wykres pozytywnych feromonów
        ax1.scatter(self.problem.coordinates[:, 0], self.problem.coordinates[:, 1],
                    c='blue', s=100, alpha=0.7, zorder=3)

        # Dodaj numery miast
        for i in range(self.n):
            ax1.annotate(str(i), (self.problem.coordinates[i, 0], self.problem.coordinates[i, 1]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, color='white',
                         weight='bold')

        # Rysuj linie dla najsilniejszych pozytywnych feromonów
        max_pos_strength = top_pos[0][0] if top_pos else 1
        for strength, i, j in top_pos:
            x_coords = [self.problem.coordinates[i, 0], self.problem.coordinates[j, 0]]
            y_coords = [self.problem.coordinates[i, 1], self.problem.coordinates[j, 1]]
            line_width = (strength / max_pos_strength) * 5 + 1  # Grubość linii proporcjonalna do siły
            ax1.plot(x_coords, y_coords, 'g-', linewidth=line_width, alpha=0.7, zorder=2)

        ax1.set_title(f'Top {top_positive} najsilniejszych pozytywnych feromonów')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)

        # Wykres negatywnych feromonów
        ax2.scatter(self.problem.coordinates[:, 0], self.problem.coordinates[:, 1],
                    c='blue', s=100, alpha=0.7, zorder=3)

        # Dodaj numery miast
        for i in range(self.n):
            ax2.annotate(str(i), (self.problem.coordinates[i, 0], self.problem.coordinates[i, 1]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8, color='white',
                         weight='bold')

        # Rysuj linie dla najsilniejszych negatywnych feromonów
        max_neg_strength = top_neg[0][0] if top_neg else 1
        for strength, i, j in top_neg:
            x_coords = [self.problem.coordinates[i, 0], self.problem.coordinates[j, 0]]
            y_coords = [self.problem.coordinates[i, 1], self.problem.coordinates[j, 1]]
            line_width = (strength / max_neg_strength) * 5 + 1  # Grubość linii proporcjonalna do siły
            ax2.plot(x_coords, y_coords, 'r-', linewidth=line_width, alpha=0.7, zorder=2)

        ax2.set_title(f'Top {top_negative} najsilniejszych negatywnych feromonów')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_pheromone_heatmaps(self):
        """Mapy ciepła dla macierzy feromonów"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Mapa ciepła dla pozytywnych feromonów
        im1 = ax1.imshow(self.tau_pos, cmap='Greens', interpolation='nearest')
        ax1.set_title('Mapa feromonów pozytywnych')
        ax1.set_xlabel('Miasto j')
        ax1.set_ylabel('Miasto i')
        plt.colorbar(im1, ax=ax1)

        # Mapa ciepła dla negatywnych feromonów
        im2 = ax2.imshow(self.tau_neg, cmap='Reds', interpolation='nearest')
        ax2.set_title('Mapa feromonów negatywnych')
        ax2.set_xlabel('Miasto j')
        ax2.set_ylabel('Miasto i')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    np.random.seed(41)
    coords = np.random.rand(50, 2)

    problem1 = TSPProblem(coords)  # For classic ACO
    problem2 = TSPProblem(coords)  # For dual-pheromone ACO\

    from ant_colony import AntColonyOptimization  # Assuming the classic ACO class is in this file

    classic_aco = AntColonyOptimization(
        problem=problem1,
        n_ants=50,
        n_iterations=10,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.3,
        q=100.0
    )

    classic_aco.run()
    best_classic = classic_aco.result()
    print("\n[Classic ACO]")
    print("Best Tour Length:", best_classic.objectives[0])
    problem1.plot_solution(best_classic)

    # algorithm = DualPheromoneACO(
    #     problem=problem2,
    #     n_ants=10,
    #     n_iterations=100,
    #     alpha=1.0,
    #     beta=2.0,
    #     gamma=2.0,
    #     rho_pos=0.3,
    #     rho_neg=0.05,
    #     n_reinforce=2,
    #     elite_boost=True,
    #     exploration_rate=0.1
    # )
    # algorithm.init_progress()
    #
    # while not algorithm.stopping_condition_is_met():
    #     algorithm.step()
    #     algorithm.update_progress()
    #     if algorithm.iteration % 10 == 0:  # Print every 10 iterations
    #         print(
    #             f"Iteration {algorithm.iteration}, Best: {algorithm._best_solution.objectives[0] if algorithm._best_solution else 'None'}")

    dual_aco = DualPheromoneACO(
        problem=problem2,
        n_ants=50,
        n_iterations=10,
        alpha=1.0,
        beta=2.0,
        gamma=2.0,
        rho_pos=0.3,
        rho_neg=0.05,
        n_reinforce=10,
        elite_boost=True,
        exploration_rate=0.05
    )

    dual_aco.init_progress()

    while not dual_aco.stopping_condition_is_met():
        dual_aco.step()
        dual_aco.update_progress()

    best_dual = dual_aco.result()
    print("\n[Dual-Pheromone ACO]")
    print("Best Tour Length:", best_dual.objectives[0])
    problem2.plot_solution(best_dual)

    dual_aco.plot_convergence()

    # result = algorithm.result()
    # print("Najlepsza trasa:", result.variables)
    # print("Długość trasy:", result.objectives[0])
    # problem2.plot_solution(result)
    #
    # algorithm.plot_convergence()
    # algorithm.plot_pheromone_matrices()
    # algorithm.plot_pheromone_city_map(20, 20)
    # algorithm.plot_pheromone_heatmaps()

    print("\n--- Comparison ---")
    print(f"Classic ACO Tour Length      : {best_classic.objectives[0]:.4f}")
    print(f"Dual-Pheromone ACO Tour Length: {best_dual.objectives[0]:.4f}")

    if best_classic.objectives[0] < best_dual.objectives[0]:
        print("Classic ACO found the shorter tour.")
    else:
        print("Dual-Pheromone ACO found the shorter tour.")

    plt.figure(figsize=(10, 5))
    plt.plot(classic_aco.best_objectives, label="Classic ACO")
    plt.plot(dual_aco.convergence_history, label="Dual-Pheromone ACO")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    dual_aco.plot_pheromone_matrices()
    dual_aco.plot_pheromone_city_map(20, 20)
