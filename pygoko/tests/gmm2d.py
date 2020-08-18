import pygoko
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt


def grab_samples(x_mean, count):
    mean = np.zeros([2],dtype=np.float32)
    mean[0] = x_mean
    cov = np.diag(np.ones([2],dtype=np.float32))
    return np.random.multivariate_normal(mean,cov,count).astype(np.float32)

data = grab_samples(0,1000)



for i in range(6):
    test_data = grab_samples(float(i)/3,50)


    fig, ax = plt.subplots()
    ax.scatter(data[:,0],data[:,1],color="orange")
    ax.scatter(test_data[:,0],test_data[:,1],color="blue")
    ax.set_xlim((-1.6,3.1))
    ax.set_ylim((-1.6,1.6))
    fig.set_size_inches(14.00, 7.00)
    fig.savefig(f"gaussian_vis_{i}.png", bbox_inches='tight')







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

