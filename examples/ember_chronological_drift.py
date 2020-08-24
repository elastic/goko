#!/usr/bin/env python

"""
This is an example that tracks chronological drift in the ember dataset. We train on the ember dataset on data before 2018-07,
and then run everything through it. There's a massive increase in the total KL-div after the cutoff, so this does detect a 
shift in the dataset.
"""

import os
import pandas as pd
import ember
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pygoko import CoverTree


def main():
    prog = "ember_drift_calc"
    descr = "Train an ember model from a directory with raw feature files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
    args = parser.parse_args()

    training_data, all_data, X_month = sort_ember_dataset(datadir = args.datadir, split_date = "2018-07")
    
    # Build the tree
    tree = CoverTree()
    tree.set_leaf_cutoff(50)
    tree.set_scale_base(1.5)
    tree.set_min_res_index(0)
    tree.fit(training_data)

    # Gather a baseline
    prior_weight = 1.0
    observation_weight = 1.3
    # 0 sets the window to be infinite, otherwise the "dataset" you're computing against is only the last `window_size` elements
    window_size = 5000   
    # We don't use this, our sequences are windowed so we only compute the KL Div on (at most) the last window_size elements
    sequence_len = 800000
    # Actually computes the KL div this often. All other values are linearly interpolated between these sample points.
    # It's too slow to calculate each value and this is accurate enough.
    sample_rate = 10
    # Gets the mean and variance over this number of simulated sequence. 
    sequence_count = 50
    
    '''
    We gather a baseline object. When you feed the entire dataset the covertree was created from to itself, 
    you will get a non-zero KL-Div on any node that is non-trivial. This process will weight the node's posterior Dirichlet distribution,
    multiplying the internal weights by (prior_weight + observation_weight). This posterior distribution has a lower variance than the prior and   
    the expected KL-divergence between the unknown distributions we're modeling is thus non-zero.

    This slowly builds up, but we expect a non-zero KL-div over the nodes as we feed in-distribution data in. This object estimates that, and
    allows us to normalize this natural variance away. 
    '''
    baseline = tree.kl_div_dirichlet_baseline(
        prior_weight,
        observation_weight,
        window_size,  
        sequence_count,
        sample_rate)
    goko_divs = {}

    """
    This is the actual object that computes the KL Divergence statistics between the samples we feed in and the new samples. 

    Internally, it is an evidence hashmap containing categorical distributions, and a queue of paths. 
    The sample's path is computed, we then push it onto the queue and update the evidence by incrementing the correct buckets 
    in the evidence hashmap. If the queue is full, we pop off the oldest path and decrement the correct paths in the queue.
    """
    run_tracker = tree.kl_div_dirichlet(
        prior_weight,
        observation_weight,
        window_size)

    total_kl_div = []

    for i,datum in enumerate(all_data):
        run_tracker.push(datum)
        if i % 500 == 0:
            goko_divs[i] = normalize(baseline,run_tracker.stats())
            total_kl_div.append(goko_divs[i]['moment1_nz'])


    fig, ax = plt.subplots()
    ax.plot(list(range(0,len(all_data),500)),total_kl_div)
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('Sample Timestamp')
    tick_len = 0
    cutoff_len = 0
    tick_locations = []
    dates = [d for d in X_month.keys()]
    for date in dates:
        if date == "2018-07":
            cutoff_len = tick_len
        tick_len += len(X_month[date])
        tick_locations.append(tick_len)
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(dates)
    ax.axvline(x=cutoff_len, linewidth=4, color='r')
    fig.tight_layout()
    fig.savefig("drift.png", bbox_inches='tight')
    plt.show()
    plt.close()

def normalize(baseline,stats):
    """
    Grabs the mean and variance from the baseline and normalizes the stats object passed in by subtracting 
    the norm and dividing by the standard deviation.
    """
    basesline_stats = baseline.stats(stats["sequence_len"])
    normalized = {}
    for k in basesline_stats.keys():
        n = (stats[k]-basesline_stats[k]["mean"])
        if basesline_stats[k]["var"] > 0:
            n = n/np.sqrt(basesline_stats[k]["var"])
        normalized[k] = n
    return normalized

def sort_ember_dataset(datadir,split_date):
    """
    Opens the dataset and creates a training dataset consisting of everything before the split date. 

    Returns the training dataset and all data
    """
    X, _  = ember.read_vectorized_features(datadir,"train")
    metadata = pd.read_csv(os.path.join(datadir, "train_metadata.csv"), index_col=0)
    dates = list(set(metadata['appeared']))
    dates.sort()

    X_month = {k:X[metadata['appeared'] == k] for k in dates}

    training_dates = [d for d in dates if d < split_date]
    all_dates = [d for d in dates]

    training_data = np.concatenate([X_month[k] for k in training_dates])
    training_data = np.ascontiguousarray(training_data)

    all_data = np.concatenate([X_month[k] for k in all_dates])
    all_data = np.ascontiguousarray(all_data)

    return training_data, all_data, X_month


if __name__ == '__main__':
    main()
