
import pygoko
import numpy as np
from math import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)


def show2D(tree, data):

    top = tree.top_scale()
    bottom = tree.bottom_scale()
    print(top, bottom)
    for j in range(top, bottom, -1):
        patches = []
        lines = []
        centers = []
        layer = tree.layer(j)
        width = layer.radius() / 2
        _, centers = layer.centers()
        for c in centers:
            patches.append(mpatches.Circle(c, 2 * width, color="Blue"))
        for i in range(j, top):
            parent_layer = tree.layer(i + 1)
            point_indexes, center_points = parent_layer.centers()
            for point_index, c in zip(point_indexes, center_points):
                if not parent_layer.is_leaf(point_index):
                    for child in parent_layer.child_points(point_index):
                        lines.append(
                            mlines.Line2D(
                                [c[0], child[0]], [c[1], child[1]], color="blue"
                            )
                        )

                for singleton in parent_layer.singleton_points(point_index):
                    lines.append(
                        mlines.Line2D(
                            [c[0], singleton[0]], [c[1], singleton[1]], color="orange"
                        )
                    )

        fig, ax = plt.subplots()
        centers = np.array(centers)

        ax.set_xlim((-1.6, 1.6))
        ax.set_ylim((-1.6, 1.6))

        collection = PatchCollection(patches, match_original=True, alpha=0.2)
        for line in lines:
            ax.add_line(line)
        ax.add_collection(collection)
        ax.scatter(data[:, 0], data[:, 1], color="orange")
        ax.scatter(centers[:, 0], centers[:, 1], color="red")

        ax.axis("off")
        fig.set_size_inches(10.00, 10.00)
        fig.savefig(f"2d_vis_{j:2d}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    numPoints = 120
    data = pi * (2 * np.random.rand(numPoints, 1) - 0.5)
    data = [np.cos(data).reshape(-1, 1), np.cos(data) * np.sin(data).reshape(-1, 1)]
    data = np.concatenate(data, axis=1).astype(np.float32)

    tree = pygoko.CoverTree()
    tree.set_leaf_cutoff(0)
    tree.set_scale_base(2.0)
    tree.fit(data)

    show2D(tree, data)
