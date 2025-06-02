# data_simulator_from_dag

Simulates synthetic data from a DAG where each node is randomly assigned as either **binary** or **continuous**. The structure of the DAG is provided as a `networkx.DiGraph`, and the simulator generates data using random parameterizations.

---

## 🚀 Features

- Accepts any DAG via `networkx.DiGraph`
- Random binary or continuous node types
- Parameter logging (weights, bias, noise) for each node

---
## 🔧 Installation

1. Clone the repository:
```python
git clone https://github.com/YOUR_USERNAME/dag_simulator.git

```

## 🧪 Usage

```python
import networkx as nx
from simulator import DAGSimulator

# Define DAG
G = nx.DiGraph()
G.add_edges_from([("X1", "Y"), ("X2", "Y"), ("X1", "Z")])

# Simulate
sim = DAGSimulator(G, n_samples=1000, seed=42)
result = sim.simulate()

# Access simulated data
df = result["data"]
```

## 🧬 Mathematical Specification

Let G = (V, E) be a DAG where:
- V is the set of nodes (variables)
- E is a subset of V × V: the set of directed edges between variables
- Each node X_i ∈ V is either binary or continuous
- Pa(X_i) ⊆ V: the set of parent nodes of X_i

### 1. Exogenous Variables (No Parents)
If Pa(X_i) is empty:

- If X_i is **continuous**:
  - X_i ~ Normal(0, 1)

- If X_i is **binary**:
  - X_i ~ Bernoulli(0.5)

### 2. Endogenous Variables (With Parents)
Let the set of parents be Pa(X_i) = {X_j1, ..., X_jk}. Define:

- Linear predictor:
  - L_i = b_i + sum(w_j * X_j for j in Pa(X_i)), where all w_j ≠ 0 and b_i ∈ ℝ

- Noise for continuous nodes:
  - ε_i ~ Normal(0, σ_i^2)

Then:

- If X_i is **binary**:
  - P(X_i = 1 | Pa(X_i)) = sigmoid(L_i) = 1 / (1 + exp(-L_i))
  - X_i ~ Bernoulli(sigmoid(L_i))

- If X_i is **continuous**:
  - X_i = L_i + ε_i

### Parameter Set per Node
- `type`: "binary" or "continuous"
- `weights`: {w_j}, with all w_j ≠ 0
- `bias`: b_i
- `noise_std`: σ_i, only for continuous nodes

### Output Format
The simulator returns a dictionary with:
- `data`: Simulated pandas DataFrame
- `node_types`: Dictionary of node types
- `parametrization`: Node-wise parameter specs
- `dag`: The original input networkx.DiGraph
