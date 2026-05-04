import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_mnist1d_examples(
    url="/home/christian/FRMDL/mnist1d/mnist1d_data.pkl",
    num_rows=2,
    num_cols=5,
):
    with open(url, "rb") as f:
        dataset = pickle.load(f)

    idxs = [np.flatnonzero(dataset["y"] == digit)[0] for digit in range(10)]

    xs = dataset["x"][idxs]
    ys = dataset["y"][idxs]
    t = dataset["t"]

    x_pad = 0.3
    x_min = xs.min() - x_pad
    x_max = xs.max() + x_pad

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )

    for ax, x, y in zip(axes.ravel(), xs, ys):
        ax.plot(x, t, linewidth=2)

        ax.set_title(f"label = {y}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(t.max(), t.min())
        ax.set_aspect("equal", adjustable="box")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


plot_mnist1d_examples(url="/home/christian/FRMDL/mnist1d/mnist1d_data.pkl")