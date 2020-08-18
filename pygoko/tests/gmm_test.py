import pygoko
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt


def grab_samples(x_mean, count):
    mean = np.zeros([100],dtype=np.float32)
    mean[0] = x_mean
    cov = np.diag(np.concatenate([np.ones([10],dtype=np.float32),0.001*np.ones([90],dtype=np.float32)]))
    return np.random.multivariate_normal(mean,cov,count).astype(np.float32)

data = grab_samples(0,100000)
tree = pygoko.CoverTree()
tree.set_leaf_cutoff(100)
tree.set_scale_base(1.5)
tree.set_min_res_index(-30)
tree.fit(data)

print("============= KL Divergence =============")
prior_weight = 1.0
observation_weight = 1.1
window_size = 1000
sequence_len = 1000
sample_rate = 25
sequence_count = 32
baseline = tree.kl_div_dirichlet_baseline(prior_weight,
    observation_weight,
    sequence_len,
    sequence_count,
    window_size,
    sample_rate)
for i in range(0,sequence_len,sample_rate):
    print(baseline.stats(i))


def unpack_stats(dataframe,stats, baseline):
    for k in baseline.keys():
        normalized = (stats[k]-baseline[k]["mean"])
        if baseline[k]["var"] > 0:
            normalized/np.sqrt(baseline[k]["var"])
        dataframe[k].append(normalized)

def cumulate(dataframes):
    cumulation = defaultdict(list)
    for dataframe in dataframes:
        for k,v in dataframe.items():
            cumulation[k].append(v)

    cumulation = {k: np.stack(v) for k,v in cumulation.items()}
    cumulation = {k: np.mean(v, axis=0) for k,v in cumulation.items()}
    return cumulation


only_cumulative = []
for j in range(2):
    run_tracker = tree.kl_div_dirichlet(
            prior_weight,
            observation_weight,
            window_size)
    run_only_tracker = tree.kl_div_dirichlet(
            prior_weight,
            observation_weight,
            window_size)
    mid_run = defaultdict(list)
    only_run = defaultdict(list)

    for i in range(100):
        test_data = grab_samples(0,sequence_len)

        test_data = grab_samples(float(i)/50,sequence_len)
        for x in test_data:
            run_only_tracker.push(x)
        unpack_stats(only_run,run_only_tracker.stats(),baseline.stats(sequence_len))
    only_cumulative.append(only_run)

print("==== Only ====")
only_cumulative = cumulate(only_cumulative)
print(only_cumulative)

fig, ax = plt.subplots()

ax.plot(np.linspace(0,2,100),only_cumulative["moment1_nz"])
ax.set_ylabel('KL Divergence')
ax.set_xlabel('Distance between mean of Multinomial in 100d')
fig.tight_layout()
fig.savefig("GaussianDrift.png", bbox_inches='tight')
plt.show()
plt.close()



'''
fig, ax = plt.subplots()

ax.plot(np.linspace(0,2,100),
ax.fill_between(
    x[10:], 
    runs["normal"]["expected_runs"][f'mean_p_stddev_{col_name}'][10:], 
    runs["normal"]["expected_runs"][f'mean_n_stddev_{col_name}'][10:], 
    color="green",
    alpha=0.5
)
attack = np.stack([run[col_name] for run in runs["normal"]["attack_runs"][:10]]).T
ax1.plot(x[10:], attack[10:], 'r', color="red", label='Attack Sequence')

normal = np.stack([run[col_name] for run in runs["normal"]["normal_runs"][:10]]).T
ax1.plot(x[10:], normal[10:], 'r', color="blue", label='Normal Sequence')
ax1.set_title("Binomial Interest")
ax1.set_xlim(10,400)

ax2.plot(x[10:],runs["uniform"]["expected_runs"][f"mean_{col_name}"][10:], color="green",
    label='Uniform Baseline')
ax2.fill_between(
    x[10:], 
    runs["uniform"]["expected_runs"][f'mean_p_stddev_{col_name}'][10:], 
    runs["uniform"]["expected_runs"][f'mean_n_stddev_{col_name}'][10:], 
    color="green",
    alpha=0.5
)
attack = np.stack([run[col_name] for run in runs["uniform"]["attack_runs"][:10]]).T
ax2.plot(x[10:], attack[10:], 'r', color="red", label='Attack Sequence')

normal = np.stack([run[col_name] for run in runs["uniform"]["normal_runs"][:10]]).T
ax2.plot(x[10:], normal[10:], 'r', color="blue", label='Normal Sequence')
ax2.set_title("Uniform Interest")
ax2.set_xlim(10,400)
plt.yscale("log")
fig.set_size_inches(12.00, 5.00)
fig.set_dpi(2000)
fig.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc='best')
fig.savefig(filename, bbox_inches='tight')
plt.close()
'''

