# 🪼 Jellyfish Search Optimization (JSO)

This repository implements both the **Original** and **Modified** versions of the Jellyfish Search Optimizer (JSO), a bio-inspired metaheuristic optimization algorithm modeled on the behavior of jellyfish in oceans.

It includes:
- 📌 A standard version using passive and active motion phases
- 🚀 An enhanced version using **Levy Flight** and **adaptive alpha control**
- 📈 Convergence visualization and 2D search trajectory plotting

---

## 🚀 Demo: Run the Optimizer

Run the main file:

```bash
python main.py
The script will:

Optimize the Rastrigin function

Display convergence curve

Plot 2D search history

🔬 Benchmark Functions Supported
✅ Classic Test Functions (for quick testing)
Sphere

Rastrigin

Rosenbrock

Ackley

Griewank

✅ CEC Benchmark Suites (results provided)
CEC 2014

CEC 2017

CEC 2020

CEC 2022

📄 See results/cec_*.md for summaries and docs/ for full result reports.

📊 Features
✅ Passive & Active movement modeling

✅ Bound-aware movement via np.clip()

✅ Levy Flight for long-range search

✅ Adaptive alpha: improves exploitation over time

✅ Visual tracking of convergence and search path

🧪 Example Objective Functions

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

## 📊 Results Summary

The **Modified JSO** demonstrates significant improvement in convergence accuracy and stability compared to the original version across all benchmark suites (CEC 2014, 2017, 2020, 2022) and standard test functions (e.g., Sphere, Rastrigin).  
📁 Detailed outputs, performance comparisons, convergence graphs, and implementation snapshots are available in the attached project ZIP file and `results/` folder.

👥 Contributors
Name	Role
Aashi P. Kumar	UI/UX + AI Developer (Lead)
Ayush Singla	Algorithm Developer
Krishna Madaan	Research + Implementation