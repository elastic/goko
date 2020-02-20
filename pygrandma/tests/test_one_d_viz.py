

import pygrandma

import numpy as np
from one_d_viz import show1D


data = np.array([[0.499], [0.48], [-0.49], [0.0]],dtype=np.float32)

tree = pygrandma.PyGrandma()
tree.set_scale_base(2)
tree.set_cutoff(0)
tree.fit(data)
show1D(tree,data)