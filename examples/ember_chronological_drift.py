    #!/usr/bin/env python

import os
import pandas as pd
import ember
import argparse
import numpy as np

import matplotlib.pyplot as plt

from pygoko import CoverTree

def split(metadata,X,Y):
    unique_dates = list(set(metadata['appeared']))

    X_sorted = {k:X[metadata['appeared'] == k] for k in unique_dates}
    y_sorted = {k:Y[metadata['appeared'] == k] for k in unique_dates}

    return X_sorted, y_sorted

def normalize(baseline,stats):
    basesline_stats = baseline.stats(stats["sequence_len"])
    normalized = {}
    for k in basesline_stats.keys():
        n = (stats[k]-basesline_stats[k]["mean"])
        if basesline_stats[k]["var"] > 0:
            n = n/np.sqrt(basesline_stats[k]["var"])
        normalized[k] = n
    return normalized

def main():
    prog = "ember_drift_calc"
    descr = "Train an ember model from a directory with raw feature files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
    args = parser.parse_args()

    split_date = "2018-07"


    # Gather the data and sort it into order
    train_X, train_y  = ember.read_vectorized_features(args.datadir,"train")
    test_X, test_y  = ember.read_vectorized_features(args.datadir,"test")

    train_metadata = pd.read_csv(os.path.join(args.datadir, "train_metadata.csv"), index_col=0)
    test_metadata = pd.read_csv(os.path.join(args.datadir, "test_metadata.csv"), index_col=0)

    train_X_month, train_y_month = split(train_metadata,train_X,train_y)
    test_X_month, test_y_month = split(test_metadata,test_X,test_y)

    print(train_X_month.keys())
    print(test_X_month.keys())

    dates = list(train_X_month.keys())
    dates.sort()

    training_dates = [d for d in dates if d < split_date]
    all_dates = [d for d in dates]

    training_data = np.concatenate([train_X_month[k] for k in training_dates])
    training_data = np.ascontiguousarray(training_data)

    all_data = np.concatenate([train_X_month[k] for k in all_dates])
    all_data = np.ascontiguousarray(all_data)

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
    # 
    sequence_len = 800000
    # Actually computes the KL div this often. All other values are linearly interpolated between these sample points
    sample_rate = 5000
    # Averages over this number of sasmples 
    sequence_count = 50
    
    baseline = tree.kl_div_dirichlet_baseline(prior_weight,
        observation_weight,
        sequence_len,
        sequence_count,
        window_size,
        sample_rate)

    goko_divs = {}


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
    tick_locations = []
    for date in all_dates:
        tick_len += len(train_X_month[date])
        tick_locations.append(tick_len)
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(all_dates)
    fig.tight_layout()
    fig.savefig("drift.png", bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
