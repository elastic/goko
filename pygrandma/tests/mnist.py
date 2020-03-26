import numpy as np
from sklearn.neighbors import KDTree
import pygrandma
import pandas as pd

data = np.memmap("../../data/mnist.dat", dtype=np.float32)
data = data.reshape([-1,784])

tree = pygrandma.PyGrandma()
tree.set_cutoff(0)
tree.set_scale_base(1.3)
tree.set_resolution(-30)
tree.fit(data)

print(tree.knn(data[0],5))

for i,layer in enumerate(tree.layers()):
    print(f"On layer {layer.scale_index()}")
    if i < 2:
        for node in layer.nodes():
            print(f"\tNode {node.address()} mean: {node.cover_mean().mean()}")

print("============= TRACE =============")
trace = tree.dry_insert(data[59999])
for address in trace:
    node = tree.node(address)
    mean = node.cover_mean()
    if mean is not None:
        print(f"\tNode {node.address()}, mean: {mean.mean()}")
    else:
        print(f"\tNode {node.address()}, MEAN IS BROKEN")

print("============= KL Divergence =============")
kl_tracker = tree.kl_div_tracker()
for x in data[:100]:
    kl_tracker.push(x)

for kl,address in kl_tracker.all_kl():
    print(kl,address)