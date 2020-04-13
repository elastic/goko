import numpy as np
from sklearn.neighbors import KDTree

tree = KDTree(data, leaf_size=2)
dist, ind = tree.query(data[:10], k=5)
print(dist[0])
print(ind[0])

dist, ind = tree.query(np.zeros([1,784],dtype=np.float32), k=5)
print(dist[0])
print(ind[0])

nbrs = {"d0":dist[:,0],
        "d1":dist[:,1],
        "d2":dist[:,2],
        "d3":dist[:,3],
        "d4":dist[:,4],
        "i0":ind[:,0],
        "i1":ind[:,1],
        "i2":ind[:,2],
        "i3":ind[:,3],
        "i4":ind[:,4],}

csv = pd.DataFrame(nbrs)
csv.to_csv("../../data/mnist_nbrs.csv")