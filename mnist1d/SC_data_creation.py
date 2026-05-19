import pickle
import numpy as np
import matplotlib.pyplot as plt


IN_PKL = "/home/christian/FRMDL_toy_dataset/mnist1d/mnist1d_data.pkl"
#OUT_PKL = "/home/christian/FRMDL_toy_dataset/mnist1d/mnist1d_data_including_marker.pkl"
OUT_PKL = "/home/christian/FRMDL_toy_dataset/mnist1d/mnist1d_data_including_marker_reversed.pkl"


# Marker positions in the 1D sequence
# MNIST-1D usually has length 40
MARKER_POS_0_TO_4 = 6
MARKER_POS_5_TO_9 = 25

MARKER_POS_0_TO_4 = 25      # uncomment for reversed case
MARKER_POS_5_TO_9 = 6       # uncomment for reversed case

def add_group_marker(dataset):
    """
    Adds marker information to the dataset without modifying the original signal.

    Original fields remain unchanged:
        x
        x_test

    New fields added:
        marker_mask
        marker_mask_test
        marker_idx
        marker_idx_test
    """

    marked = dataset.copy()

    x = marked["x"]
    y = marked["y"]
    x_test = marked["x_test"]
    y_test = marked["y_test"]

    seq_len = x.shape[1]

    if not (0 <= MARKER_POS_0_TO_4 < seq_len):
        raise ValueError("MARKER_POS_0_TO_4 is outside the sequence length.")

    if not (0 <= MARKER_POS_5_TO_9 < seq_len):
        raise ValueError("MARKER_POS_5_TO_9 is outside the sequence length.")

    marker_idx = np.where(
        y <= 4,
        MARKER_POS_0_TO_4,
        MARKER_POS_5_TO_9,
    )

    marker_idx_test = np.where(
        y_test <= 4,
        MARKER_POS_0_TO_4,
        MARKER_POS_5_TO_9,
    )

    marker_mask = np.zeros_like(x)
    marker_mask_test = np.zeros_like(x_test)

    marker_mask[np.arange(x.shape[0]), marker_idx] = 1.0
    marker_mask_test[np.arange(x_test.shape[0]), marker_idx_test] = 1.0

    marked["marker_idx"] = marker_idx
    marked["marker_idx_test"] = marker_idx_test
    marked["marker_mask"] = marker_mask
    marked["marker_mask_test"] = marker_mask_test

    marked["marker_info"] = {
        "description": "Separate marker mask. Original x and x_test are unchanged.",
        "labels_0_to_4_marker_position": MARKER_POS_0_TO_4,
        "labels_5_to_9_marker_position": MARKER_POS_5_TO_9,
        "marker_value": 1.0,
        "x_unchanged": True,
        "x_test_unchanged": True,
    }

    return marked


def plot_examples_with_group_marker(dataset, num_rows=2, num_cols=5):
    idxs = [np.flatnonzero(dataset["y"] == digit)[0] for digit in range(10)]

    xs = dataset["x"][idxs]
    ys = dataset["y"][idxs]
    t = dataset["t"]
    marker_idxs = dataset["marker_idx"][idxs]

    x_pad = 0.8
    x_min = xs.min() - x_pad
    x_max = xs.max() + x_pad

    # Put visual marker to the right of the signal,
    # so it is not part of the curve.
    marker_x_visual = x_max - 0.2

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )

    for ax, x, y, marker_idx in zip(axes.ravel(), xs, ys, marker_idxs):
        ax.plot(x, t, linewidth=2)

        # This marker is visualized separately from the line.
        ax.scatter(
            marker_x_visual,
            t[marker_idx],
            s=90,
            marker="s",
            zorder=5,
        )

        ax.set_title(f"label = {int(y)}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(t.max(), t.min())
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    with open(IN_PKL, "rb") as f:
        dataset = pickle.load(f)

    original_x = dataset["x"].copy()
    original_x_test = dataset["x_test"].copy()

    marked_dataset = add_group_marker(dataset)

    assert np.array_equal(original_x, marked_dataset["x"])
    assert np.array_equal(original_x_test, marked_dataset["x_test"])

    with open(OUT_PKL, "wb") as f:
        pickle.dump(marked_dataset, f, protocol=3)

    print(f"Saved marked dataset to: {OUT_PKL}")
    print("x unchanged:", np.array_equal(original_x, marked_dataset["x"]))
    print("x_test unchanged:", np.array_equal(original_x_test, marked_dataset["x_test"]))
    print("marker_mask shape:", marked_dataset["marker_mask"].shape)
    print("marker_mask_test shape:", marked_dataset["marker_mask_test"].shape)

    print()
    print("Labels 0-4 marker position:", MARKER_POS_0_TO_4)
    print("Labels 5-9 marker position:", MARKER_POS_5_TO_9)

    plot_examples_with_group_marker(marked_dataset)


if __name__ == "__main__":
    main()