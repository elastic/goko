import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import pandas as pd
from sklearn.neighbors import KDTree
import os

# Base MNIST transform for easy access. 
# The Yaml files are often messed with, these are the base files.
mnist_yaml = '''
---
cutoff: 5
resolution: -10
scale_base: 2
data_path: ../data/mnist.dat
labels_path: ../data/mnist.csv
count: 60000
data_dim: 784
labels_dim: 10
in_ram: True
'''

mnist_complex_yaml = '''
---
cutoff: 0
resolution: -20
scale_base: 1.3
use_singletons: true
verbosity: 0
data_path: ../data/mnist.dat
labels_path: ../data/mnist.csv
count: 60000
data_dim: 784
in_ram: True
schema:
  'y': i32
  name: string
'''

metaFile = open("data/mnist.yml","wb")
metaFile.write(mnist_yaml.encode('utf-8'))
metaFile.close()
metaFile = open("data/mnist_complex.yml","wb")
metaFile.write(mnist_complex_yaml.encode('utf-8'))
metaFile.close()

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype(np.float32)
x_train = x_train.reshape(-1, 28*28)
dataFile = open("mnist.dat", "wb")
for x in x_train:
    dataFile.write(x.tobytes())
dataFile.close()
y_bools = [y%2 == 0 for y in y_train]
y_str = [str(y) for y in y_train]

df = pd.DataFrame({"y":y_train,"even":y_bools,"name":y_str})
df.index.rename('index', inplace=True)
df.to_csv('mnist.csv')

# KNN data for tests
data = np.memmap("mnist.dat", dtype=np.float32)
data = data.reshape([-1,784])

tree = KDTree(data, leaf_size=2)
dist, ind = tree.query(data[:100], k=5)

dist, ind = tree.query(np.zeros([1,784],dtype=np.float32), k=5)

nbrs = {"d0":dist[:,0],
        "d1":dist[:,1],
        "d2":dist[:,2],
        "d3":dist[:,3],
        "d4":dist[:,4],
        "i0": ind[:,0],
        "i1": ind[:,1],
        "i2": ind[:,2],
        "i3": ind[:,3],
        "i4": ind[:,4],}

csv = pd.DataFrame(nbrs)
csv.to_csv("mnist_nbrs.csv")