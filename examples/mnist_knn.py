#!/usr/bin/env python

"""
This example shows loading from a YAML file. You can specify all the parameters in the yaml file. 
Thiss 
"""

import numpy as np
from pygoko import CoverTree

tree = CoverTree()
tree.load_yaml_config("data/mnist_complex.yml")
tree.fit()

point = np.zeros([784], dtype=np.float32)

"""
This is a standard KNN, returning the 5 nearest nbrs.
"""

print(tree.knn(point,5))

"""
This is the KNN that ignores singletons, the outliers attached to each node and the leftover indexes on the leaf
are ignored.
"""

print(tree.routing_knn(point,5))

"""
This returns the indexes of the nodes along the path. We can then ask for the label summaries of 
each node along the path.
"""
path = tree.path(point)
print(path)

print("Summary of the labels of points covered by the node at address")
for dist, address in path:
    node = tree.node(address)
    label_summary = node.label_summary()
    print(f"Address {address}: Summary: {label_summary}")

