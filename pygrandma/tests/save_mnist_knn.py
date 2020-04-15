import numpy as np
from sklearn.neighbors import KDTree

tree = KDTree(data, leaf_size=2)
dist, ind = tree.query(data[:10], k=5)
print(dist[0])
print(ind[0])

dist, ind = tree.query(np.zeros([1,784],dtype=np.float32), k=5)
print(dist[0])
print(ind[0])

with open('../../data/mnist_nbrs.csv','w') as outfile:
        outfile.write('d0,d1,d2,d3,d4,i0,i1,i2,i3,i4\n')
        for d,i in zip(dist,ind):
                outfile.write(f'{d[0]},{d[1]},{d[2]},{d[3]},{d[4]},{i[0]},{i[1]},{i[2]},{i[3]},{i[4]}\n')