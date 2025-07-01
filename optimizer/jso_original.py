import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class JellySearchOptimization:
    def __init__(self, objective_function, n_jellyfish=100, dimensions=2, lb=-5.0, ub=5.0,
                 max_iterations=2000, c1=1.0, c2=2.0):
        """
        Initialize the Jelly Search Optimization algorithm.

        Parameters:
        -----------
        objective_function : callable
            The function to be minimized
        n_jellyfish : int
            Number of jellyfish (population size)
        dimensions : int
            Number of dimensions in the search space
        lb : float or array-like
            Lower bounds of the search space
        ub : float or array-like
            Upper bounds of the search space
        max_iterations : int
            Maximum number of iterations
        c1 : float
            Control parameter for passive motion
        c2 : float
            Control parameter for active motion
        """
        self.objective_function = objective_function
        self.n_jellyfish = n_jellyfish
        self.dimensions = dimensions
        self.lb = np.ones(dimensions) * lb if np.isscalar(lb) else np.array(lb)
        self.ub = np.ones(dimensions) * ub if np.isscalar(ub) else np.array(ub)
        self.max_iterations = max_iterations
        self.c1 = c1
        self.c2 = c2

        # Initialize best solution
        self.best_position = None
        self.best_fitness = float('inf')

        # For tracking progress
        self.convergence_curve = np.zeros(max_iterations)
        self.all_positions = []
        self.all_fitness = []

    def initialize_population(self):
        """Initialize the jellyfish population randomly within bounds."""
        return self.lb + np.random.random((self.n_jellyfish, self.dimensions)) * (self.ub - self.lb)

    def evaluate_population(self, population):
        """Evaluate the fitness of each jellyfish in the population."""
        fitness = np.zeros(self.n_jellyfish)
        for i in range(self.n_jellyfish):
            fitness[i] = self.objective_function(population[i])
        return fitness

    def passive_motion(self, population, fitness):
        """Implement passive motion of jellyfish (based on ocean current)."""
        # Find current best position
        best_idx = np.argmin(fitness)
        local_best = population[best_idx].copy()

        # Update the global best if needed
        if fitness[best_idx] < self.best_fitness:
            self.best_position = local_best.copy()
            self.best_fitness = fitness[best_idx]

        # Ocean current effect (move towards the best solution)
        new_population = population.copy()
        for i in range(self.n_jellyfish):
            # Random movement component
            r1 = np.random.random(self.dimensions)
            # Move towards best solution
            new_population[i] = population[i] + self.c1 * r1 * (self.best_position - population[i])

        # Ensure bounds
        new_population = np.maximum(new_population, self.lb)
        new_population = np.minimum(new_population, self.ub)

        return new_population

    def active_motion(self, population, fitness):
        """Implement active motion of jellyfish (individual movement)."""
        new_population = population.copy()

        for i in range(self.n_jellyfish):
            # Select random jellyfish (different from current)
            j = i
            while j == i:
                j = np.random.randint(0, self.n_jellyfish)

            # If the neighbor is better, move towards it
            if fitness[j] < fitness[i]:
                r2 = np.random.random(self.dimensions)
                new_population[i] = population[i] + self.c2 * r2 * (population[j] - population[i])
            else:  # Otherwise, random movement
                r3 = np.random.random(self.dimensions)
                new_population[i] = population[i] + self.c2 * (r3 - 0.5)

        # Ensure bounds
        new_population = np.maximum(new_population, self.lb)
        new_population = np.minimum(new_population, self.ub)

        return new_population

    def optimize(self, verbose=False):
        """Run the optimization process."""
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)

        # Initialize best solution
        best_idx = np.argmin(fitness)
        self.best_position = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iterations):
            # Store current positions for visualization
            self.all_positions.append(population.copy())
            self.all_fitness.append(fitness.copy())

            # Passive motion (movement due to ocean current)
            population = self.passive_motion(population, fitness)
            fitness = self.evaluate_population(population)

            # Active motion (individual movement)
            population = self.active_motion(population, fitness)
            fitness = self.evaluate_population(population)

            # Update best solution found
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_position = population[best_idx].copy()
                self.best_fitness = fitness[best_idx]

            # Store the best fitness value
            self.convergence_curve[iteration] = self.best_fitness

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best fitness: {self.best_fitness}")

        return self.best_position, self.best_fitness

    def plot_convergence(self):
        """Plot the convergence curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_iterations + 1), self.convergence_curve)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

    def plot_search_history_2d(self):
        """Plot the search history for 2D problems."""
        if self.dimensions != 2:
            print("This plot is only available for 2D problems")
            return

        plt.figure(figsize=(10, 8))

        # Plot the function contour if possible
        if hasattr(self, 'plot_function_2d'):
            self.plot_function_2d(plt)

        # Plot initial population
        initial_pop = self.all_positions[0]
        plt.scatter(initial_pop[:, 0], initial_pop[:, 1], color='blue', marker='o', alpha=0.5, label='Initial Population')

        # Plot final population
        final_pop = self.all_positions[-1]
        plt.scatter(final_pop[:, 0], final_pop[:, 1], color='red', marker='x', alpha=0.8, label='Final Population')

        # Plot best solution
        plt.scatter(self.best_position[0], self.best_position[1], color='green', marker='*', s=200, label='Best Solution')

        # Plot trajectory of a few jellyfish
        n_samples = min(5, self.n_jellyfish)
        for i in range(n_samples):
            traj_x = [self.all_positions[j][i, 0] for j in range(0, len(self.all_positions), max(1, len(self.all_positions)//10))]
            traj_y = [self.all_positions[j][i, 1] for j in range(0, len(self.all_positions), max(1, len(self.all_positions)//10))]
            plt.plot(traj_x, traj_y, 'k--', alpha=0.3)

        plt.title('Search History')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
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
    n_jellyfish = 50
    max_iterations = 100

    # Choose objective function
    objective_function = rastrigin_function

    # Create and run optimizer
    jso = JellySearchOptimization(
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

        plt_obj.contour(X, Y, Z, 50, cmap='viridis', alpha=0.5)

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