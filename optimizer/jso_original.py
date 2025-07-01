import numpy as np

class JellyfishOriginal:
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

    def optimize(self):
        for t in range(self.max_iter):
            c = 1 - (t / self.max_iter)  # Time control function
            for i in range(self.num_agents):
                current_score = self.obj_func(self.positions[i])
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = self.positions[i].copy()

                if np.random.rand() < c:
                    # Ocean current (exploration)
                    self.positions[i] += np.random.rand() * (self.best_position - self.positions[i])
                else:
                    # Swarm movement (exploitation)
                    if np.random.rand() < 0.5:
                        # Passive
                        j = np.random.randint(self.num_agents)
                        self.positions[i] += np.random.rand() * (self.positions[j] - self.positions[i])
                    else:
                        # Active
                        self.positions[i] += np.random.rand() * (self.best_position - self.positions[i])

                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
        return self.best_position, self.best_score