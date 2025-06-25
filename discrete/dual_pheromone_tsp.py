from matplotlib import pyplot as plt

import numpy as np
import random
from jmetal.core.solution import PermutationSolution
from jmetal.core.algorithm import Algorithm
from jmetal.util.observable import DefaultObservable
from typing import List

from discrete.tsp_problem import TSPProblem

import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ExplorationDecayType(Enum):
    """Typy strategii zmniejszania exploration_rate"""
    NONE = "none"  # Brak zmniejszania
    LINEAR = "linear"  # Liniowe zmniejszanie
    EXPONENTIAL = "exponential"  # Wykładnicze zmniejszanie
    INVERSE = "inverse"  # Odwrotnie proporcjonalne


@dataclass
class PheromoneConfig:
    """Konfiguracja parametrów feromonowych"""
    min_pheromone: float = 1e-6  # Minimalna wartość feromonu (wcześniej hardcoded)
    pheromone_reward_factor: float = 100.0  # Czynnik nagrody feromonowej
    negative_pheromone_scaling: float = 0.1  # Skalowanie negatywnego feromonu


@dataclass
class ExplorationConfig:
    """Konfiguracja strategii eksploracji"""
    initial_rate: float = 0.1  # Początkowy wskaźnik eksploracji
    final_rate: float = 0.01  # Końcowy wskaźnik eksploracji
    decay_type: ExplorationDecayType = ExplorationDecayType.LINEAR
    decay_start_iteration: int = 0  # Od której iteracji zaczynać zmniejszanie
    decay_factor: float = 0.95  # Czynnik dla decay wykładniczego (0 < decay_factor < 1)


class DualPheromoneACO(Algorithm):
    """
    Algorytm Dual Pheromone Ant Colony Optimization dla problemu TSP.

    Wykorzystuje pozytywne i negatywne feromony do kierowania eksploracją
    i eksploatacją w przestrzeni rozwiązań.
    """

    def get_name(self) -> str:
        return "Dual Pheromone ACO"

    def __init__(self,
                 problem: TSPProblem,
                 n_ants: int = 10,
                 n_iterations: int = 100,
                 alpha: float = 1.0,  # Wpływ pozytywnego feromonu
                 beta: float = 2.0,  # Wpływ heurystyki odległości
                 gamma: float = 2.0,  # Wpływ negatywnego feromonu
                 rho_pos: float = 0.1,  # Współczynnik parowania pozytywnego feromonu
                 rho_neg: float = 0.1,  # Współczynnik parowania negatywnego feromonu
                 n_reinforce: int = 2,  # Liczba najlepszych/najgorszych rozwiązań do wzmocnienia
                 elite_boost: bool = False,  # Czy stosować dodatkowe wzmocnienie dla najlepszego
                 exploration_rate: float = 0.1,  # Prawdopodobieństwo eksploracji (używane gdy exploration_config=None)
                 pheromone_config: Optional[PheromoneConfig] = None,
                 exploration_config: Optional[ExplorationConfig] = None):

        super().__init__()
        self.problem = problem
        self.n_ants = n_ants
        self.max_iterations = n_iterations

        # Parametry algorytmu
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho_pos = rho_pos
        self.rho_neg = rho_neg
        self.n_reinforce = n_reinforce
        self.elite_boost = elite_boost

        # Konfiguracja eksploracji
        self.exploration_config = exploration_config
        if exploration_config is None:
            # Używamy stałego exploration_rate jeśli nie podano konfiguracji
            self.current_exploration_rate = exploration_rate
        else:
            self.current_exploration_rate = exploration_config.initial_rate

        # Konfiguracja feromonów
        self.pheromone_config = pheromone_config or PheromoneConfig()

        # Inicjalizacja struktur danych
        self.n = problem.number_of_variables()
        self._initialize_pheromones()
        self._precompute_distance_matrix()

        # Stan algorytmu
        self.observable = DefaultObservable()
        self._reset_state()

    def _initialize_pheromones(self) -> None:
        """Inicjalizuje macierze feromonów"""
        self.tau_pos = np.ones((self.n, self.n), dtype=np.float32)
        self.tau_neg = np.zeros((self.n, self.n), dtype=np.float32)

    def _precompute_distance_matrix(self) -> None:
        """Prekompiluje macierz odległości dla optymalizacji"""
        coords = self.problem.coordinates
        n = len(coords)
        self.distance_matrix = np.zeros((n, n), dtype=np.float32)

        # Wykorzystanie wektoryzacji NumPy dla szybszego obliczania
        for i in range(n):
            diff = coords - coords[i]  # Broadcasting
            distances = np.linalg.norm(diff, axis=1)
            self.distance_matrix[i] = distances

        # Unikanie dzielenia przez zero
        np.fill_diagonal(self.distance_matrix, np.inf)

    def _reset_state(self) -> None:
        """Resetuje stan algorytmu"""
        self.iteration = 0
        self.evaluations = 0
        self._best_solution = None
        self.convergence_history = []
        # Reset exploration rate do wartości początkowej
        if self.exploration_config is not None:
            self.current_exploration_rate = self.exploration_config.initial_rate

    def create_initial_solutions(self) -> List[PermutationSolution]:
        """Tworzy początkowe rozwiązania"""
        return [self._construct_solution() for _ in range(self.n_ants)]

    def evaluate(self, solution_list: List[PermutationSolution]) -> List[PermutationSolution]:
        """Ewaluuje listę rozwiązań"""
        return [self.problem.evaluate(solution) for solution in solution_list]

    def init_progress(self) -> None:
        """Inicjalizuje postęp algorytmu"""
        self._reset_state()
        self.observable.notify_all(self.observable_data())

    def stopping_condition_is_met(self) -> bool:
        """Sprawdza czy warunek stopu został spełniony"""
        return self.iteration >= self.max_iterations

    def step(self) -> None:
        """Wykonuje jeden krok algorytmu"""
        # Konstruuj i ewaluuj rozwiązania
        solutions = [self._construct_solution() for _ in range(self.n_ants)]
        evaluated_solutions = self.evaluate(solutions)
        self.evaluations += len(evaluated_solutions)

        # Aktualizuj najlepsze rozwiązanie
        self._update_best_solution(evaluated_solutions)

        # Aktualizuj feromony
        self._update_pheromones(evaluated_solutions)

        # Aktualizuj exploration rate
        self._update_exploration_rate()

        # Aktualizuj stan
        self.iteration += 1
        if self._best_solution:
            self.convergence_history.append(self._best_solution.objectives[0])

    def _update_best_solution(self, solutions: List[PermutationSolution]) -> None:
        """Aktualizuje najlepsze rozwiązanie"""
        for solution in solutions:
            if (self._best_solution is None or
                    solution.objectives[0] < self._best_solution.objectives[0]):
                self._best_solution = solution

    def update_progress(self) -> None:
        """Aktualizuje postęp dla obserwatorów"""
        self.observable.notify_all(self.observable_data())

    def observable_data(self) -> Dict:
        """Zwraca dane dla obserwatorów"""
        return {
            "ITERATION": self.iteration,
            "EVALUATIONS": self.evaluations,
            "COMPUTING_TIME": 0.0,
            "SOLUTIONS": [self._best_solution] if self._best_solution else []
        }

    def result(self) -> PermutationSolution:
        """Zwraca najlepsze znalezione rozwiązanie"""
        return self._best_solution

    def _construct_solution(self) -> PermutationSolution:
        """
        Konstruuje pojedyncze rozwiązanie używając strategii dual pheromone
        """
        tour = []
        n = self.n
        current_city = np.random.randint(n)
        tour.append(current_city)

        visited = np.zeros(n, dtype=bool)
        visited[current_city] = True

        while len(tour) < n:
            unvisited_mask = ~visited
            unvisited_indices = np.flatnonzero(unvisited_mask)

            if np.random.random() < self.current_exploration_rate:
                next_city = self._exploration_selection(current_city, unvisited_mask, unvisited_indices)
            else:
                next_city = self._exploitation_selection(current_city, unvisited_mask, unvisited_indices)

            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city

        return self._create_solution_from_tour(tour)

    def _exploration_selection(self, current: int, unvisited_mask: np.ndarray,
                               unvisited_indices: np.ndarray) -> int:
        """Selekcja w trybie eksploracji - preferuje niski negatywny feromon"""
        tau_neg_values = self.tau_neg[current, unvisited_mask]
        attractiveness = 1.0 / (self.pheromone_config.min_pheromone + tau_neg_values)
        probabilities = attractiveness / attractiveness.sum()
        return np.random.choice(unvisited_indices, p=probabilities)

    def _exploitation_selection(self, current: int, unvisited_mask: np.ndarray,
                                unvisited_indices: np.ndarray) -> int:
        """Selekcja w trybie eksploatacji - używa pozytywnego feromonu i heurystyki"""
        tau_pos_values = self.tau_pos[current, unvisited_mask] ** self.alpha
        eta_values = (1.0 / self.distance_matrix[current, unvisited_mask]) ** self.beta
        psi_values = (1.0 + self.tau_neg[current, unvisited_mask]) ** self.gamma

        probabilities = (tau_pos_values * eta_values) / psi_values
        probabilities /= probabilities.sum()

        return np.random.choice(unvisited_indices, p=probabilities)

    def _create_solution_from_tour(self, tour: List[int]) -> PermutationSolution:
        """Tworzy obiekt rozwiązania z trasy"""
        solution = PermutationSolution(
            number_of_variables=self.n,
            number_of_objectives=1
        )
        solution.variables = tour
        return solution

    def _update_pheromones(self, solutions: List[PermutationSolution]) -> None:
        """
        Aktualizuje macierze feromonów na podstawie jakości rozwiązań
        """
        # Parowanie feromonów
        self.tau_pos *= (1 - self.rho_pos)
        self.tau_neg *= (1 - self.rho_neg)

        # Sortowanie rozwiązań według jakości
        solutions.sort(key=lambda s: s.objectives[0])

        good_solutions = solutions[:self.n_reinforce]
        bad_solutions = solutions[-self.n_reinforce:]

        # Pozytywne wzmocnienie
        self._apply_positive_reinforcement(good_solutions)

        # Negatywne wzmocnienie
        self._apply_negative_reinforcement(good_solutions, bad_solutions)

    def _apply_positive_reinforcement(self, good_solutions: List[PermutationSolution]) -> None:
        """Stosuje pozytywne wzmocnienie dla dobrych rozwiązań"""
        for solution in good_solutions:
            delta = self.pheromone_config.pheromone_reward_factor / solution.objectives[0]
            self._reinforce_solution_edges(solution, self.tau_pos, delta)

        # Elite boost dla najlepszego rozwiązania
        if self.elite_boost and good_solutions:
            best_solution = good_solutions[0]
            elite_delta = self.pheromone_config.pheromone_reward_factor / best_solution.objectives[0]
            self._reinforce_solution_edges(best_solution, self.tau_pos, elite_delta)

    def _apply_negative_reinforcement(self, good_solutions: List[PermutationSolution],
                                      bad_solutions: List[PermutationSolution]) -> None:
        """Stosuje negatywne wzmocnienie dla złych krawędzi"""
        good_edges = self._extract_edges_from_solutions(good_solutions)
        bad_edge_stats = self._analyze_bad_edges(bad_solutions, good_edges)

        if not bad_edge_stats:
            return

        max_count = max(stats['count'] for stats in bad_edge_stats.values())

        for edge, stats in bad_edge_stats.items():
            frequency_factor = stats['count'] / max_count
            avg_quality = stats['total_quality'] / stats['count']

            delta = (frequency_factor * (1 / avg_quality) *
                     self.pheromone_config.negative_pheromone_scaling)

            a, b = edge
            self.tau_neg[a, b] += delta
            self.tau_neg[b, a] += delta

    def _extract_edges_from_solutions(self, solutions: List[PermutationSolution]) -> Set[Tuple[int, int]]:
        """Wyciąga krawędzie z listy rozwiązań"""
        edges = set()
        for solution in solutions:
            for i in range(self.n):
                a, b = solution.variables[i], solution.variables[(i + 1) % self.n]
                edges.add((min(a, b), max(a, b)))
        return edges

    def _analyze_bad_edges(self, bad_solutions: List[PermutationSolution],
                           good_edges: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], Dict]:
        """Analizuje statystyki złych krawędzi"""
        bad_edge_stats = {}

        for solution in bad_solutions:
            solution_quality = self.pheromone_config.pheromone_reward_factor / solution.objectives[0]

            for i in range(self.n):
                a, b = solution.variables[i], solution.variables[(i + 1) % self.n]
                edge = (min(a, b), max(a, b))

                # Tylko krawędzie nieobecne w dobrych rozwiązaniach
                if edge not in good_edges:
                    if edge not in bad_edge_stats:
                        bad_edge_stats[edge] = {'count': 0, 'total_quality': 0}

                    bad_edge_stats[edge]['count'] += 1
                    bad_edge_stats[edge]['total_quality'] += solution_quality

        return bad_edge_stats

    def _update_exploration_rate(self) -> None:
        """
        Aktualizuje exploration_rate zgodnie z wybraną strategią
        """
        if self.exploration_config is None:
            return  # Brak zmian - używamy stałego exploration_rate

        config = self.exploration_config

        # Sprawdź czy już czas zacząć zmniejszanie
        if self.iteration < config.decay_start_iteration:
            return

        # Oblicz postęp (0.0 - 1.0)
        effective_iteration = self.iteration - config.decay_start_iteration
        effective_max_iterations = self.max_iterations - config.decay_start_iteration

        if effective_max_iterations <= 0:
            progress = 1.0
        else:
            progress = min(1.0, effective_iteration / effective_max_iterations)

        # Zastosuj odpowiednią strategię decay
        if config.decay_type == ExplorationDecayType.NONE:
            # Bez zmian
            pass
        elif config.decay_type == ExplorationDecayType.LINEAR:
            self.current_exploration_rate = (
                    config.initial_rate -
                    progress * (config.initial_rate - config.final_rate)
            )
        elif config.decay_type == ExplorationDecayType.EXPONENTIAL:
            # Wykładnicze zmniejszanie używając decay_factor
            self.current_exploration_rate = max(
                config.final_rate,
                config.initial_rate * (config.decay_factor ** effective_iteration)
            )
        elif config.decay_type == ExplorationDecayType.INVERSE:
            # Odwrotnie proporcjonalne: rate = initial_rate / (1 + k * iteration)
            # k dobrane tak, aby przy końcowej iteracji osiągnąć final_rate
            if effective_max_iterations > 0:
                k = (config.initial_rate / config.final_rate - 1) / effective_max_iterations
                self.current_exploration_rate = max(
                    config.final_rate,
                    config.initial_rate / (1 + k * effective_iteration)
                )

        # Zabezpieczenie przed wartościami poza zakresem
        self.current_exploration_rate = max(0.0, min(1.0, self.current_exploration_rate))

    def get_current_exploration_rate(self) -> float:
        """Zwraca aktualny exploration_rate"""
        return self.current_exploration_rate

    def _reinforce_solution_edges(self, solution: PermutationSolution,
                                  pheromone_matrix: np.ndarray, delta: float) -> None:
        """Wzmacnia krawędzie w danym rozwiązaniu"""
        for i in range(self.n):
            a, b = solution.variables[i], solution.variables[(i + 1) % self.n]
            pheromone_matrix[a, b] += delta
            pheromone_matrix[b, a] += delta

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
        n_iterations=200,
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

    pheromone_config = PheromoneConfig(
        min_pheromone=1e-6,
        pheromone_reward_factor=100,
        negative_pheromone_scaling=0.1
    )

    exploration_config = ExplorationConfig(
        initial_rate=0.1,
        final_rate=0.0,
        decay_type=ExplorationDecayType.LINEAR
    )

    dual_aco = DualPheromoneACO(
        problem=problem2,
        n_ants=50,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        gamma=2.0,
        rho_pos=0.3,
        rho_neg=0.05,
        n_reinforce=10,
        elite_boost=True,
        pheromone_config=pheromone_config,
        exploration_config=exploration_config,
    )

    dual_aco.init_progress()

    while not dual_aco.stopping_condition_is_met():
        dual_aco.step()
        dual_aco.update_progress()

    best_dual = dual_aco.result()
    print("\n[Dual-Pheromone ACO]")
    print("Best Tour Length:", best_dual.objectives[0])
    problem2.plot_solution(best_dual)

    # dual_aco.plot_convergence()

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
    # dual_aco.plot_pheromone_city_map(20, 20)
