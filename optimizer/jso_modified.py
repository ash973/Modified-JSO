import numpy as np

class JellyfishModified:
    def __init__(self, obj_func, num_agents, max_iter, dim, lb, ub):
        self.obj_func = obj_func
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.positions = np.random.uniform(lb, ub, (num_agents, dim))
        self.best_position = None
        self.best_score = float('inf')

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v)**(1 / beta)
        return step

    def optimize(self):
        for t in range(self.max_iter):
            alpha = 1 - (t / self.max_iter)  # adaptive switch control
            for i in range(self.num_agents):
                score = self.obj_func(self.positions[i])
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.positions[i].copy()

                if np.random.rand() < alpha:
                    # Passive (towards best)
                    self.positions[i] += np.random.rand() * (self.best_position - self.positions[i])
                else:
                    # Active (Levy flight)
                    self.positions[i] += self.levy_flight()

                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

        return self.best_position, self.best_score