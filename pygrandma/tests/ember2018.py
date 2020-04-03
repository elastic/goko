import numpy as np
from sklearn.neighbors import KDTree
import pygrandma
import pandas as pd

tree = pygrandma.PyGrandma()
tree.fit_yaml("../data/ember_complex.yml")

print(tree.knn(tree.data_point(0),5))


print("============= TRACE =============")
trace = tree.dry_insert(tree.data_point(59999))
for address in trace:
    node = tree.node(address)
    mean = node.cover_mean()
    if mean is not None:
        print(f"\tNode {node.address()}, mean: {mean.mean()}")
    else:
        print(f"\tNode {node.address()}, MEAN IS BROKEN")

print("============= KL Divergence =============")
normal_stats = tree.kl_div_sgd_basestats(0.005,0.8)
for i,stats in enumerate(normal_stats):
    print(stats)
print("============= KL Divergence Normal Use =============")
kl_tracker = tree.kl_div_sgd(0.005,0.8)
for i in range(200):
    kl_tracker.push(tree.data_point(i))
    print(kl_tracker.stats())


print("============= KL Divergence Attack =============")

kl_attack_tracker = tree.kl_div_sgd(0.005,0.8)
for i in range(100):
    kl_attack_tracker.push(tree.data_point(i))
    print(kl_attack_tracker.stats())
for i in range(100):
    kl_attack_tracker.push(tree.data_point(0))
    print(kl_attack_tracker.stats())