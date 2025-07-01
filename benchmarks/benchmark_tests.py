import numpy as np
def sphere(x):
    return sum(x**2)

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(i**2 - A * np.cos(2 * np.pi * i)) for i in x])

def rosenbrock(x):
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    x = np.array(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def griewank(x):
    x = np.array(x)
    return 1 + sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))