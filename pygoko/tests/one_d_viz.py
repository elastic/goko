import pygoko

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib import cm

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)


def show1D(
    tree,
    data,
    final_layer=-1000,
    color="conditional",
    tracker=None,
    test_set=None,
):
    patches = []
    lines = []
    x_center = []
    y_center = []

    ungraphed_nodes = [tree.root()]

    while len(ungraphed_nodes) > 0:
        node = ungraphed_nodes.pop()
        (si, pi) = node.address()
        x = data[pi, 0]
        y = 1 + 0.1 * si
        width = 2 ** si
        x_center.append(x)
        y_center.append(y)

        singleton_len = float(len(node.singletons_indexes()))

        if si < final_layer:
            break
        child_probs = node.children_probs()
        singleton_probs =  [(a,v) for a,v in child_probs if a is None][0]
        child_probs = [(a,v) for a,v in child_probs if a is not None]
        for i, (child_address, child_prob) in enumerate(child_probs):
            (csi, cpi) = child_address
            y_next = 1 + 0.1 * csi
            x_next = data[cpi][0]
            child = tree.node(child_address)
            if color == "conditional":
                lines.append(
                    mlines.Line2D(
                        [x, x_next], [y, y - 0.05], color=cmap(child_prob), linewidth=4
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, y_next],
                        color=cmap(child_prob),
                        linewidth=4,
                    )
                )
            elif color == "total":
                coverage_count = float(child.coverage_count())
                lines.append(
                    mlines.Line2D(
                        [x, x_next],
                        [y, y - 0.05],
                        color=cmap(coverage_count / len(data)),
                        linewidth=4,
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, y_next],
                        color=cmap(coverage_count / len(data)),
                        linewidth=4,
                    )
                )
            elif "post_prob" in color:
                tracker_probs = tracker.marginal_posterior_probs((si, pi))
                tracker_probs = [(a,v) for a,v in tracker_probs if a is not None]
                my_color = tracker_probs[i][1]
                lines.append(
                    mlines.Line2D(
                        [x, x_next], [y, y - 0.05], color=cmap(my_color), linewidth=4
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, y_next],
                        color=cmap(my_color),
                        linewidth=4,
                    )
                )
            elif color == "evidence":
                evidence = tracker.evidence((si, pi))
                if evidence is not None:
                    tracker_probs, _ = evidence
                    my_color = tracker_probs[i][1]
                else:
                    my_color = 0
                lines.append(
                    mlines.Line2D(
                        [x, x_next], [y, y - 0.05], color=cmap(my_color), linewidth=4
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, y_next],
                        color=cmap(my_color),
                        linewidth=4,
                    )
                )

            ungraphed_nodes.append(child)
        for cpi in node.singletons_indexes():
            x_next = data[cpi][0]
            if color == "conditional":
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, 0.0],
                        color=cmap(singleton_probs),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x, x_next],
                        [y, y - 0.05],
                        color=cmap(singleton_probs),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
            elif color == "total":
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, 0.0],
                        color=cmap(singleton_len / len(data)),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x, x_next],
                        [y, y - 0.05],
                        color=cmap(singleton_len / len(data)),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
            elif "post_prob" in color:
                tracker_probs = tracker.marginal_posterior_probs((si, pi))
                tracker_probs = [(a,v) for a,v in tracker_probs if a is None]
                my_color = tracker_probs[0][1]
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, 0.0],
                        color=cmap(my_color),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x, x_next],
                        [y, y - 0.05],
                        color=cmap(my_color),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
            elif color == "evidence":
                evidence = tracker.evidence((si, pi))
                if evidence is not None:
                    _, my_color = evidence
                else:
                    my_color = 0
                lines.append(
                    mlines.Line2D(
                        [x_next, x_next],
                        [y - 0.05, 0.0],
                        color=cmap(my_color),
                        linewidth=4,
                    )
                )
                lines.append(
                    mlines.Line2D(
                        [x, x_next], [y, y - 0.05], color=cmap(my_color), linewidth=4
                    )
                )
        if node.is_leaf():
            patches.append(
                mpatches.Rectangle(
                    [x - width, y - 0.05], 2 * width, 0.08, ec="none", color="orange"
                )
            )
            if color == "conditional":
                lines.append(
                    mlines.Line2D(
                        [x, x],
                        [y, 0.0],
                        color=cmap(singleton_probs),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
            elif color == "total":
                lines.append(
                    mlines.Line2D(
                        [x, x],
                        [y, 0.0],
                        color=cmap(1.0 / len(data)),
                        linewidth=2,
                        alpha=0.5,
                    )
                )
            elif "post_prob" in color:
                my_color = tracker.marginal_posterior_probs((si, pi))[0][1]
                lines.append(
                    mlines.Line2D(
                        [x, x], [y, 0.0], color=cmap(my_color), linewidth=2, alpha=0.5
                    )
                )
            elif color == "evidence":
                evidence = tracker.evidence((si, pi))
                if evidence is not None:
                    _, my_color = evidence
                else:
                    my_color = 0
                lines.append(
                    mlines.Line2D([x, x], [y, 0.0], color=cmap(my_color), linewidth=4)
                )

        else:
            patches.append(
                mpatches.Rectangle(
                    [x - width, y - 0.05], 2 * width, 0.08, ec="none", color="blue"
                )
            )
    fig, ax = plt.subplots()

    ax.set_xlim((-1.01, 1.01))
    ax.set_ylim((-0.05, 1.2))

    if test_set is not None:
        for c in test_set:

            path = tree.path(c)
            csi, cpi = path[-1][0]
            x = data[cpi][0]
            y = 1 + 0.1 * csi
            x_next = c[0]
            lines.append(
                mlines.Line2D([x, x_next], [y, y - 0.05], color=cmap(my_color), linewidth=2)
            )
            lines.append(
                mlines.Line2D([x_next, x_next], [y - 0.05, 0.0], color=cmap(my_color), linewidth=2)
            )
        ax.scatter(
            test_set[:, 0].reshape((test_set.shape[0],)),
            np.zeros([test_set.shape[0]]),
            color="red",
            zorder=2.5,
        )
    
    collection = PatchCollection(patches, match_original=True, alpha=0.3)
    for line in lines:
        ax.add_line(line)
    ax.scatter(x_center, y_center)
    ax.add_collection(collection)
    cax = fig.add_axes([0.95, 0.05, 0.01, 0.9])
    ax.scatter(
        data[:, 0].reshape((data.shape[0],)),
        np.zeros([data.shape[0]]),
        color="orange",
        alpha=1,
        zorder=2.5,
    )
    
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical"
    )

    ax.axis("off")
    fig.set_size_inches(14.00, 7.00)
    fig.savefig(f"1d_vis_{color}.png", bbox_inches="tight")


if __name__ == "__main__":
    numPoints = 21
    numTest = 10

    data1 = np.random.normal(loc=0.5, scale=0.2, size=(numPoints // 3,)).reshape(
        numPoints // 3, 1
    )
    data2 = np.random.normal(loc=-0.5, scale=0.1, size=(2 * numPoints // 3,)).reshape(
        2 * numPoints // 3, 1
    )
    data = np.concatenate([data1, data2], axis=0)
    data = np.concatenate([data, np.zeros([numPoints, 1])], axis=1).astype(np.float32)

    test_set = np.random.normal(loc=0.5, scale=0.2, size=(10,)).reshape(10, 1)
    test_set = np.concatenate([test_set, np.zeros([10, 1])], axis=1).astype(np.float32)

    tree = pygoko.CoverTree()
    tree.set_scale_base(2)
    tree.set_leaf_cutoff(0)
    tree.fit(data)
    path = tree.path(np.array([0.5], dtype=np.float32))

    run_tracker = tree.tracker(0)
    for i, t in enumerate(test_set):
        run_tracker.push(t)
        show1D(
            tree,
            data,
            color=f"post_prob_{i}",
            tracker=run_tracker,
            test_set=test_set[:i],
        )
    show1D(tree, data, color="total")
    show1D(tree, data)
