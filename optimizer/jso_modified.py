import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class OptimizedJSO:
    def __init__(self, objective_function, n_jellyfish=50, dimensions=2, lb=-5.0, ub=5.0,
                 max_iterations=2000, c1=0.4, c2=2, beta=2.5):
        self.objective_function = objective_function
        self.n_jellyfish = n_jellyfish
        self.dim = dimensions
        self.lb = np.ones(dimensions) * lb if np.isscalar(lb) else np.array(lb)
        self.ub = np.ones(dimensions) * ub if np.isscalar(ub) else np.array(ub)
        self.max_iterations = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.beta = beta

        self.best_position = None
        self.best_fitness = float("inf")
        self.convergence_curve = np.zeros(max_iterations)
        self.all_positions = []

    def initialize_population(self):
        return self.lb + np.random.rand(self.n_jellyfish, self.dim) * (self.ub - self.lb)

    def evaluate(self, population):
        return np.array([self.objective_function(ind) for ind in population])

    def levy_flight(self):
        numerator = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        denominator = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        sigma = (abs(numerator / denominator)) ** (1 / self.beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / self.beta))
        return step

    def update_position(self, current, target, step_size):
        move = step_size * (target - current)
        return current + move

    def optimize(self, verbose=True):
        population = self.initialize_population()
        fitness = self.evaluate(population)

        best_idx = np.argmin(fitness)
        self.best_position = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]

        for t in range(self.max_iterations):
            self.all_positions.append(population.copy())

            alpha = 1 - t / self.max_iterations  # exploitation increases over time
            new_population = population.copy()

            for i in range(self.n_jellyfish):
                r = np.random.rand()
                if r < alpha:
                    # Passive motion (toward global best)
                    r1 = np.random.rand(self.dim)
                    new_population[i] = population[i] + self.c1 * r1 * (self.best_position - population[i])
                else:
                    # Active motion (Levy-based)
                    step = self.levy_flight()
                    j = np.random.randint(0, self.n_jellyfish)
                    new_population[i] = population[i] + self.c2 * step * (population[j] - population[i])

                # Bound check
                new_population[i] = np.clip(new_population[i], self.lb, self.ub)

            new_fitness = self.evaluate(new_population)

            # Update global best
            best_idx = np.argmin(new_fitness)
            if new_fitness[best_idx] < self.best_fitness:
                self.best_position = new_population[best_idx].copy()
                self.best_fitness = new_fitness[best_idx]

            population = new_population
            fitness = new_fitness
            self.convergence_curve[t] = self.best_fitness

            if verbose and (t + 1) % 10 == 0:
                print(f"Iteration {t + 1}/{self.max_iterations}, Best fitness: {self.best_fitness:.6f}")

        return self.best_position, self.best_fitness

    def plot_convergence(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.convergence_curve)
        plt.title("Convergence Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.show()
    
    def plot_search_history_2d(self):
        if self.dim != 2:
            print("This plot is only for 2D problems.")
            return

        plt.figure(figsize=(10, 8))

        # Plot objective function contour (if it's provided)
        if hasattr(self, 'plot_function_2d'):
            self.plot_function_2d(plt)

        # Initial positions
        initial_pop = self.all_positions[0]
        plt.scatter(initial_pop[:, 0], initial_pop[:, 1], color='blue', marker='o', alpha=0.5, label='Initial Population')

        # Final positions
        final_pop = self.all_positions[-1]
        plt.scatter(final_pop[:, 0], final_pop[:, 1], color='red', marker='x', alpha=0.8, label='Final Population')

        # Best solution
        plt.scatter(self.best_position[0], self.best_position[1], color='green', marker='*', s=200, label='Best Solution')

        # Trajectories of few jellyfish
        n_samples = min(5, self.n_jellyfish)
        for i in range(n_samples):
            traj_x = [self.all_positions[j][i, 0] for j in range(0, len(self.all_positions), max(1, len(self.all_positions)//10))]
            traj_y = [self.all_positions[j][i, 1] for j in range(0, len(self.all_positions), max(1, len(self.all_positions)//10))]
            plt.plot(traj_x, traj_y, 'k--', alpha=0.3)

        plt.title('Search History (Optimized JSO)')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage
def sphere_function(x):
    """Sphere function (global minimum at origin)."""
    return np.sum(x**2)

def rosenbrock_function(x):
    """Rosenbrock function (global minimum at [1,1,...])."""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin_function(x):
    """Rastrigin function (global minimum at origin)."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Demo function
def run_jso_demo():
    # Define problem parameters
    dimensions = 2
    lb = -5.0
    ub = 5.0
    n_jellyfish = 100
    max_iterations = 100

    # Choose objective function
    objective_function = rastrigin_function

    # Create and run optimizer
    jso = OptimizedJSO(
        objective_function=objective_function,
        n_jellyfish=n_jellyfish,
        dimensions=dimensions,
        lb=lb,
        ub=ub,
        max_iterations=max_iterations
    )

    # Add method for plotting 2D function contour
    def plot_function_2d(plt_obj):
        x = np.linspace(lb, ub, 100)
        y = np.linspace(lb, ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = objective_function(np.array([X[i, j], Y[i, j]]))

        plt_obj.contour(X, Y, Z, 0, cmap='viridis', alpha=0.1)

    jso.plot_function_2d = plot_function_2d

    # Run optimization
    best_solution, best_fitness = jso.optimize(verbose=True)

    print("\nOptimization Results:")
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")

    # Visualize results
    jso.plot_convergence()
    jso.plot_search_history_2d()

if __name__ == "__main__":
    run_jso_demo()