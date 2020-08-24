'''
This produces 2 gaussians, one that is fixed at 0, and the other that moves slowly over. 
'''

import pygoko
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    # How many samples we grab from the fixed gaussian
    fixed_sample_count = 200000
    # How many samples we grab from the moving gaussian, at each timestamp
    moving_sample_count = 1000
    timestamps = np.linspace(0,2,50)

    # We treat multiply the weight vector of the Dirichlet prior by this before computing the KL-div 
    prior_weight = 1.0
    # We weight the evidence by this before we add it to the prior to get the posterior.  
    observation_weight = 1.0
    # How often we sample the KL div of the sequences
    sample_rate = 25
    # How many artificial sequences do we average over
    sequence_count = 32

    tree = build_covertree(sample_from_gaussian(0,fixed_sample_count))

    # See the ember_ chronological _drift
    baseline = tree.kl_div_dirichlet_baseline(
        prior_weight,
        observation_weight,
        moving_sample_count,  # We don't need to sample sequences longer than this
        sequence_count,
        sample_rate)



    tracking_stats = []
    for t in timestamps:
        run_only_tracker = tree.kl_div_dirichlet(
                prior_weight,
                observation_weight,
                0)
        timestamps_stats = defaultdict(list)

        for i in range(10):
            moving_data = sample_from_gaussian(t,moving_sample_count)
            for x in moving_data:
                run_only_tracker.push(x)
            unpack_stats(
                timestamps_stats,
                run_only_tracker.stats(),
                baseline.stats(moving_sample_count))
        tracking_stats.append(timestamps_stats)

    plot(timestamps,tracking_stats)

def sample_from_gaussian(x_mean, count):
    """
    Grabs count samples from a gaussian centered at [x_mean, 0, ... 0], with the identity matrix for the covariance.
    """
    mean = np.zeros([100],dtype=np.float32)
    mean[0] = x_mean
    cov = np.diag(np.concatenate([np.ones([10],dtype=np.float32),0.001*np.ones([90],dtype=np.float32)]))
    return np.random.multivariate_normal(mean,cov,count).astype(np.float32)

def build_covertree(data):
    """
    Builds a covertree on the data
    """
    tree = pygoko.CoverTree()
    tree.set_leaf_cutoff(100)
    tree.set_scale_base(1.5)
    tree.set_min_res_index(-30)
    tree.fit(data)
    return tree

def unpack_stats(dataframe,stats, baseline):
    """
    Normalizes the stats by the baseline
    """
    for k in baseline.keys():
        normalized = (stats[k]-baseline[k]["mean"])
        if baseline[k]["var"] > 0:
            normalized/np.sqrt(baseline[k]["var"])
        dataframe[k].append(normalized)

def plot(timestamps,dataframes,statistic="moment1_nz"):
    cumulation = defaultdict(list)
    for dataframe in dataframes:
        for k,v in dataframe.items():
            cumulation[k].append(v)

    cumulation = {k: np.stack(v) for k,v in cumulation.items()}
    cumulation_mean = {k: np.mean(v, axis=1) for k,v in cumulation.items()}
    fig, ax = plt.subplots()

    ax.plot(timestamps,cumulation_mean[statistic])
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('Distance between mean of Multinomial in 100d')
    fig.tight_layout()
    fig.savefig("GaussianDrift.png", bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()