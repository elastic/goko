import numpy as np
from sklearn.neighbors import KDTree
import pygoko
import pandas as pd

data = np.memmap("../../data/mnist.dat", dtype=np.float32)
data = data.reshape([-1,784])

tree = pygoko.CoverTree()
tree.load_yaml_config("../../data/mnist_complex.yml")
tree.set_leaf_cutoff(0)
tree.set_scale_base(1.2)
tree.set_min_res_index(-30)
tree.fit()

print(tree.knn(data[0],5))

for i,layer in enumerate(tree.layers()):
    print(f"On layer {layer.scale_index()}")
    if i < 2:
        for node in layer.nodes():
            print(f"\tNode {node.address()} mean: {node.cover_mean().mean()}")

print("============= TRACE =============")
trace = tree.path(data[59999])
for address in trace:
    print(address)
    node = tree.node(address[1])
    mean = node.cover_mean()
    if mean is not None:
        print(f"\tNode {node.address()}, mean: {mean.mean()}, summary: {node.label_summary()}")
    else:
        print(f"\tNode {node.address()}, MEAN IS BROKEN")

print("============= KL Divergence =============")
normal_stats = tree.kl_div_dirichlet_basestats(1.0,1.3,100,10,20)
for i,vstats in enumerate(normal_stats[:1]):
    for stats in vstats:
        print(stats)
print("============= KL Divergence Normal Use =============")
kl_tracker = tree.kl_div_dirichlet(1.0,1.3,20)
for x in data[:50]:
    kl_tracker.push(x)
    print(kl_tracker.stats())


print("============= KL Divergence Attack =============")

kl_attack_tracker = tree.kl_div_dirichlet(1.0,1.3,20)
for i in range(50):
    kl_attack_tracker.push(data[0])
    print(kl_attack_tracker.stats())
