import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X, y, coeffs = make_regression(n_samples = 10000, n_features = 1000, coef = True, noise = 100.0)
y -= y.mean()

rf = RandomForestRegressor(n_estimators = 100, max_depth = 10, random_state = 2, n_jobs = -1, verbose = 1)
rf.fit(X, y)
importances = rf.feature_importances_.tolist()

graph = nx.DiGraph()

covs = []
for i in range(np.shape(X)[1]):
    cov = abs(np.cov(X[:, i], y)[0][1])
    covs.append(cov)
    graph.add_edge(i, (np.shape(X)[1]), weight = cov)
    for j in range(np.shape(X)[1]):
        cov = abs(np.cov(X[:, i], X[:, j])[0][1])
        graph.add_edge(i, j, weight = cov)
        graph.add_edge(j, i, weight = cov)

centrality = list(nx.eigenvector_centrality(graph, weight = 'weight').values())

print()
print(np.argsort(-np.array(centrality))[:10])
print()
print(np.argsort(-np.array(covs))[:10])
print()
print(np.argsort(-np.array(importances))[:10])
print()
print(np.argsort(-coeffs)[:10])
print()
