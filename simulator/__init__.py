"""
SimulatorFromDAG
============

Simulates data from a given directed acyclic graph (DAG), where nodes are either binary or continuous.
Node types are randomly assigned. For each node, data is generated based on the types of its parents
and a specified mathematical mechanism.

Mathematical Specification
--------------------------
Let \( G = (V, E) \) be a DAG with:
- \( V \): set of nodes (variables)
- \( E \): set of directed edges
- \( \text{Pa}(X_i) \): parent nodes of \( X_i \in V \)

### 1. Exogenous Variables (no parents)
If \( \text{Pa}(X_i) = \emptyset \):
- If \( X_i \) is **continuous**:
  \[ X_i \sim \mathcal{N}(0, 1) \]
- If \( X_i \) is **binary**:
  \[ X_i \sim \text{Bernoulli}(0.5) \]

### 2. Endogenous Variables (with parents)
Let:
- \( L_i = b_i + \sum_{j \in \text{Pa}(X_i)} w_j X_j \): linear combination
- \( w_j \neq 0 \): weight for parent \( X_j \)
- \( b_i \in \mathbb{R} \): bias

Then:
- If \( X_i \) is **binary**:
  \[ P(X_i = 1 \mid \text{Pa}(X_i)) = \sigma(L_i) = \frac{1}{1 + e^{-L_i}} \]
  \[ X_i \sim \text{Bernoulli}(\sigma(L_i)) \]
- If \( X_i \) is **continuous**:
  \[ X_i = L_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2) \]

Parameters per node:
- `type`: "binary" or "continuous"
- `weights`: dict of non-zero weights \( w_j \)
- `bias`: scalar \( b_i \)
- `noise_std`: \( \sigma_i \), if continuous

Returns:
--------
A dictionary with:
- `data`: simulated DataFrame
- `node_types`: type assignment per node
- `parametrization`: generation parameters
- `dag`: original input DAG
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import expit

class SimulatorFromDAG:
    def __init__(self, graph: nx.DiGraph, n_samples=1000, seed=None):
        self.graph = graph
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.node_types = {}
        self.parametrization = {}
        self.data = pd.DataFrame(index=range(n_samples))

    def _sample_nonzero_weight(self, low=0.1, high=2.0):
        sign = self.rng.choice([-1, 1])
        return sign * self.rng.uniform(low, high)

    def _generate_nonzero_weights(self, parents):
        return {parent: self._sample_nonzero_weight() for parent in parents}

    def _assign_node_types(self):
        for node in self.graph.nodes():
            self.node_types[node] = self.rng.choice(["binary", "continuous"])

    def simulate(self):
        self._assign_node_types()
        topo_order = list(nx.topological_sort(self.graph))

        for node in topo_order:
            parents = list(self.graph.predecessors(node))
            node_type = self.node_types[node]

            if not parents:
                if node_type == "continuous":
                    self.data[node] = self.rng.normal(loc=0.0, scale=1.0, size=self.n_samples)
                    self.parametrization[node] = {
                        "type": "continuous", "dist": "normal", "mean": 0, "std": 1
                    }
                else:
                    draws = self.rng.binomial(1, 0.5, size=self.n_samples)
                    self.data[node] = draws
                    self.parametrization[node] = {
                        "type": "binary", "dist": "bernoulli", "p": 0.5," draws": draws
                    }
            else:
                weights = self._generate_nonzero_weights(parents)
                bias = self.rng.normal(0, 1)
                lin_comb = sum(weights[p] * self.data[p] for p in parents) + bias

                if node_type == "binary":
                    prob = expit(lin_comb)
                    uniform_noise = self.rng.uniform(0, 1, size=self.n_samples)
                    draws = (uniform_noise < prob).astype(int)
                    self.data[node] = draws
                    self.parametrization[node] = {
                        "type": "binary", 
                        "link": "sigmoid", 
                        "weights": weights, 
                        "bias": bias,
                        "prob": prob,
                        "uniform_noise": uniform_noise,
                        "draws": draws
                    }
                else:
                    noise_std = self.rng.uniform(0.5, 1.5)
                    noise = self.rng.normal(0, noise_std, size=self.n_samples)
                    self.data[node] = lin_comb + noise
                    self.parametrization[node] = {
                        "type": "continuous", 
                        "link": "linear", 
                        "weights": weights, 
                        "bias": bias, 
                        "noise_std": noise_std,
                        "draws": noise
                    }

        return {
            "data": self.data,
            "node_types": self.node_types,
            "parametrization": self.parametrization,
            "dag": self.graph
        }
