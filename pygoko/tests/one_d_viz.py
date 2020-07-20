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

def show1D(tree,data):
    patches = []
    lines = []
    x_center = []
    y_center = []
    top = tree.top_scale()
    bottom = tree.bottom_scale()
    for i in range(top,bottom,-1):
        layer = tree.layer(i)
        width = layer.radius()
        scale_index = layer.scale_index()

        y = 1.5 + 0.1*scale_index
        point_indexes, center_points = layer.centers()
        for my_pi, c in zip(point_indexes,center_points):
            x = c[0]
            x_center.append(x)
            y_center.append(y)
            child_addresses = layer.child_addresses(my_pi)
            if layer.is_leaf(my_pi):
                patches.append(mpatches.Rectangle([x - width, y-0.05],2*width, 0.08, ec="none",color="orange"))

                lines.append(mlines.Line2D([x,x], [y,0.0],color="blue"))

            else:
                patches.append(mpatches.Rectangle([x - width, y-0.05],2*width, 0.08, ec="none",color="blue"))
                for si,pi in child_addresses:
                    y_next = 1.5+0.1*si
                    x_next = data[pi][0]
                    lines.append(mlines.Line2D([x,x_next], [y,y-0.05],color="blue"))
                    lines.append(mlines.Line2D([x_next,x_next], [y-0.05,y_next],color="blue"))

            for c in layer.singleton_points(my_pi):
                lines.append(mlines.Line2D([c[0],c[0]], [y-0.05,0.0],color="orange"))
                lines.append(mlines.Line2D([x,c[0]], [y,y-0.05],color="orange"))

    fig, ax = plt.subplots()
    ax.set_xlim((-1.01,1.01))
    ax.set_ylim((-0.1,2.1))
    ax.scatter(data[:,0].reshape((data.shape[0],)),np.zeros([data.shape[0]]),color="orange")
    collection = PatchCollection(patches,match_original=True, alpha=0.3)
    for l in lines:
        ax.add_line(l)
    ax.scatter(x_center,y_center)
    ax.add_collection(collection)
    plt.show()
    
if __name__ == '__main__':
    numPoints = 42

    data1 = np.random.normal(loc=0.5,scale=0.2,size=(numPoints//3,)).reshape(numPoints//3,1)
    data2 = np.random.normal(loc=-0.5,scale=0.1,size=(2*numPoints//3,)).reshape(2*numPoints//3,1)
    data = np.concatenate([data1,data2],axis=0)
    data = np.concatenate([data,np.zeros([numPoints,1])],axis=1).astype(np.float32)

    tree = pygoko.CoverTree()
    tree.set_scale_base(2)
    tree.set_leaf_cutoff(0)
    tree.fit(data)
    show1D(tree,data)