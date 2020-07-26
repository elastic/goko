import numpy as np
import pygoko

tree = pygoko.CoverTree()
tree.load_yaml_config("../../data/ember_complex.yml")
tree.fit()


print(tree.knn(tree.data_point(0),5))

print("============= TRACE =============")
trace = tree.path(tree.data_point(59999))
for distance,address in trace:
    node = tree.node(address)
    mean = node.cover_mean()
    if mean is not None:
        print(f"\tNode {node.address()}, mean: {mean.mean()}")
    else:
        print(f"\tNode {node.address()}, MEAN IS BROKEN")

print("============= KL Divergence =============")
prior_weight = 1.0
observation_weight = 1.3
sequence_cap = 10
sequence_len = 20
sequence_count = 10
normal_stats = tree.kl_div_dirichlet_basestats(prior_weight,
    observation_weight,
    sequence_len,
    sequence_count,
    sequence_cap)
for i,stats in enumerate(normal_stats[0]):
    print(stats)
print("============= KL Divergence Normal Use =============")
kl_tracker = tree.kl_div_dirichlet(prior_weight,observation_weight,sequence_cap)
for i in range(200):
    kl_tracker.push(tree.data_point(i))
    print(kl_tracker.stats())


print("============= KL Divergence Attack =============")

kl_attack_tracker = tree.kl_div_dirichlet(prior_weight,observation_weight,sequence_cap)
for i in range(10):
    kl_attack_tracker.push(tree.data_point(i))
    print(kl_attack_tracker.stats())
for i in range(10):
    kl_attack_tracker.push(tree.data_point(0))
    print(kl_attack_tracker.stats())