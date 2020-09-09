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
tree.attach_svds(3,2000,1.0)

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

"""
We can also do a lookup for the paths of the points included in the training set. This is faster than a normal path 
lookup as there fewer distance calculations. 
"""

path = tree.known_path(40000)
print(path)

print("Summary of the labels of points covered by the node at address")
for dist, address in path:
    node = tree.node(address)
    label_summary = node.label_summary()
    print(f"Address {address}: Summary: {label_summary}")
    sing = node.get_singular_values()
    if sing is not None:
        print(sing[:30])


"""
We can also use the tree as a generative model. You can navigate the tree until you get to a leaf, then sample 
from a gaussian built off the singletons associated to the leaf.  
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(10, 10),
                 axes_pad=0.1, 
                 )

for ax in grid:
    sample,label = tree.sample()
    ax.imshow(sample.reshape([28,28]))

plt.show()