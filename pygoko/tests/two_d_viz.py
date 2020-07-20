import pygoko

import numpy as np
from math import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib import cm
cmap= plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            
def show2D(tree,data):
    patches = []
    lines = []
    centers = []
    top = tree.top_scale()
    bottom = tree.bottom_scale()
    print(top,bottom)
    for i in range(top,bottom,-1):
        layer = tree.layer(i)
        width = layer.radius()
        point_indexes, center_points = layer.centers()
        for pi, c in zip(point_indexes,center_points):
            patches.append(mpatches.Circle(c,2*width,color="Blue"))
            centers.append(c)
            if not layer.is_leaf(pi):
                for child in layer.child_points(pi):
                    lines.append(mlines.Line2D([c[0],child[0]], [c[1],child[1]],color="blue"))
                
            for singleton in layer.singleton_points(pi):
                lines.append(mlines.Line2D([c[0],singleton[0]], [c[1],singleton[1]],color="orange"))

    centers = np.array(centers)
    fig, ax = plt.subplots()
    ax.set_xlim((-1.1,1.1))
    ax.set_ylim((-1.1,1.1))

    cmap= plt.get_cmap("jet")
    collection = PatchCollection(patches,match_original=True,alpha= 0.05)
    for l in lines:
        ax.add_line(l)
    ax.scatter(centers[:,0],centers[:,1])
    ax.add_collection(collection)
    plt.show()

if __name__ == '__main__':
    numPoints = 120
    data = pi*(2*np.random.rand(numPoints,1) - 0.5)
    data = [np.cos(data).reshape(-1,1),np.cos(data)*np.sin(data).reshape(-1,1)]
    data = np.concatenate(data,axis=1).astype(np.float32)

    tree = pygoko.Covertree()
    tree.set_leaf_cutoff(0)
    tree.fit(data)

    show2D(tree,data)
