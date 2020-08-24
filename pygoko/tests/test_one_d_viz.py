

import pygoko

import numpy as np
from one_d_viz import show1D


data = np.array([[0.499], [0.48], [-0.49], [0.0]],dtype=np.float32)


tree = pygoko.CoverTree()
tree.set_scale_base(2)
tree.set_leaf_cutoff(0)
tree.fit(data)

print(tree.knn(tree.data_point(0),5))

print("============= KL Divergence =============")
prior_weight = 1.0
observation_weight = 1.0
window_size = 3
sequence_len = 10
sample_rate = 2
sequence_count = 1
baseline = tree.kl_div_dirichlet_baseline(prior_weight,
    observation_weight,
    sequence_len,
    sequence_count,
    window_size,
    sample_rate)
for i in range(0,sequence_len,sample_rate):
    print(baseline.stats(i))

show1D(tree,data)