from optimizer import JellyfishModified
from benchmarks import sphere, rastrigin, rosenbrock, ackley, griewank

benchmark_functions = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Ackley": ackley,
    "Griewank": griewank
}

dim = 30
lb = -5.12
ub = 5.12

for name, func in benchmark_functions.items():
    print(f"\nRunning Modified JSO on {name} Function:")
    jso = JellyfishModified(obj_func=func, num_agents=30, max_iter=100, dim=dim, lb=lb, ub=ub)
    best_position, best_score = jso.optimize()
    print("Best Score:", best_score)