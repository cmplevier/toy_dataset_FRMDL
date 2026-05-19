import pickle
import numpy as np
import matplotlib.pyplot as plt


IN_PKL = "/home/christian/FRMDL_toy_dataset/mnist1d/mnist1d_data.pkl"

OUT_PKL_NORMAL = (
    "/home/christian/FRMDL_toy_dataset/mnist1d/"
    "mnist1d_data_x_length_50_with_marker.pkl"
)

OUT_PKL_REVERSED = (
    "/home/christian/FRMDL_toy_dataset/mnist1d/"
    "mnist1d_data_x_length_50_with_marker_reversed.pkl"
)


# Original MNIST-1D is usually length 40.
# We extend it to length 50.
NEW_SEQ_LEN = 50

# Marker is placed at the final point, after a blank gap.
MARKER_POS = NEW_SEQ_LEN - 1

# Blank value in the added region.
BLANK_VALUE = 0.0


def add_group_marker(dataset, reversed_marker=False):
    """
    Adds marker information directly into x and x_test.

    Original x length:
        40

    New x length:
        50

    Structure:
        positions 0-39  : original MNIST-1D signal
        positions 40-48 : blank space
        position 49     : group marker

    Normal marker:
        labels 0-4 get marker value at the top value
        labels 5-9 get marker value at the bottom value

    Reversed marker:
        labels 0-4 get marker value at the bottom value
        labels 5-9 get marker value at the top value

    The dtype of x and x_test is preserved.
    """

    marked = dataset.copy()

    x = marked["x"]
    y = marked["y"]
    x_test = marked["x_test"]
    y_test = marked["y_test"]

    old_seq_len = x.shape[1]

    if NEW_SEQ_LEN <= old_seq_len:
        raise ValueError("NEW_SEQ_LEN must be larger than the original sequence length.")

    if not (0 <= MARKER_POS < NEW_SEQ_LEN):
        raise ValueError("MARKER_POS is outside the new sequence length.")

    # Use values from the original signal range.
    # These are explicitly cast to the original dtype.
    top_value = np.array(np.max(x), dtype=x.dtype)
    bottom_value = np.array(np.min(x), dtype=x.dtype)

    top_value_test = np.array(np.max(x_test), dtype=x_test.dtype)
    bottom_value_test = np.array(np.min(x_test), dtype=x_test.dtype)

    # Create new longer signals with the same dtype as the original x.
    x_new = np.full(
        (x.shape[0], NEW_SEQ_LEN),
        np.array(BLANK_VALUE, dtype=x.dtype),
        dtype=x.dtype,
    )

    x_test_new = np.full(
        (x_test.shape[0], NEW_SEQ_LEN),
        np.array(BLANK_VALUE, dtype=x_test.dtype),
        dtype=x_test.dtype,
    )

    # Copy original signal into the first part.
    x_new[:, :old_seq_len] = x
    x_test_new[:, :old_seq_len] = x_test

    if not reversed_marker:
        # Normal:
        # Labels 0-4: top marker value.
        # Labels 5-9: bottom marker value.
        x_new[y <= 4, MARKER_POS] = top_value
        x_new[y >= 5, MARKER_POS] = bottom_value

        x_test_new[y_test <= 4, MARKER_POS] = top_value_test
        x_test_new[y_test >= 5, MARKER_POS] = bottom_value_test

        labels_0_to_4_marker = "top"
        labels_5_to_9_marker = "bottom"
        marker_mode = "normal"

    else:
        # Reversed:
        # Labels 0-4: bottom marker value.
        # Labels 5-9: top marker value.
        x_new[y <= 4, MARKER_POS] = bottom_value
        x_new[y >= 5, MARKER_POS] = top_value

        x_test_new[y_test <= 4, MARKER_POS] = bottom_value_test
        x_test_new[y_test >= 5, MARKER_POS] = top_value_test

        labels_0_to_4_marker = "bottom"
        labels_5_to_9_marker = "top"
        marker_mode = "reversed"

    # Store original x for safety/debugging.
    marked["x_original"] = x.copy()
    marked["x_test_original"] = x_test.copy()

    # Replace x and x_test with the length-50 versions.
    marked["x"] = x_new
    marked["x_test"] = x_test_new

    # Store marker index info.
    marked["marker_idx"] = np.full(x.shape[0], MARKER_POS, dtype=np.int64)
    marked["marker_idx_test"] = np.full(x_test.shape[0], MARKER_POS, dtype=np.int64)

    # Extend t so plotting x against t still works.
    if "t" in marked:
        t = marked["t"]
        dt = t[1] - t[0]
        extra_t = t[-1] + dt * np.arange(1, NEW_SEQ_LEN - old_seq_len + 1)

        marked["t_original"] = t.copy()
        marked["t"] = np.concatenate([t, extra_t])

    marked["marker_info"] = {
        "description": "Marker added directly into x and x_test after blank appended region.",
        "marker_mode": marker_mode,
        "old_sequence_length": old_seq_len,
        "new_sequence_length": NEW_SEQ_LEN,
        "blank_region": [old_seq_len, MARKER_POS - 1],
        "marker_position": MARKER_POS,
        "labels_0_to_4_marker": labels_0_to_4_marker,
        "labels_5_to_9_marker": labels_5_to_9_marker,
        "top_value": float(top_value),
        "bottom_value": float(bottom_value),
        "blank_value": float(BLANK_VALUE),
        "x_dtype": str(x_new.dtype),
        "x_test_dtype": str(x_test_new.dtype),
    }

    return marked


def plot_examples_with_group_marker(dataset, title, num_rows=2, num_cols=5):
    idxs = [np.flatnonzero(dataset["y"] == digit)[0] for digit in range(10)]

    xs = dataset["x"][idxs]
    ys = dataset["y"][idxs]

    if "t" in dataset:
        t = dataset["t"]
    else:
        t = np.arange(xs.shape[1])

    marker_pos = dataset["marker_info"].get("marker_position", MARKER_POS)
    old_seq_len = dataset["marker_info"].get("old_sequence_length", xs.shape[1] - 10)

    x_pad = 0.8
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
        # Plot only the original MNIST-1D signal.
        # Do not include the appended blank region or marker in the line plot.
        ax.plot(x[:old_seq_len], t[:old_seq_len], linewidth=2)

        # Highlight the marker point separately.
        ax.scatter(
            x[marker_pos],
            t[marker_pos],
            s=90,
            marker="s",
            zorder=5,
        )

        # Shade the appended blank + marker region.
        if old_seq_len < len(t):
            ax.axhspan(t[old_seq_len], t[-1], alpha=0.05)

        ax.set_title(f"label = {int(y)}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(t.max(), t.min())
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def check_marked_dataset(original_dataset, marked_dataset, reversed_marker=False):
    original_x = original_dataset["x"]
    original_x_test = original_dataset["x_test"]

    y = original_dataset["y"]
    y_test = original_dataset["y_test"]

    old_seq_len = original_x.shape[1]

    # Check shape.
    assert marked_dataset["x"].shape[1] == NEW_SEQ_LEN
    assert marked_dataset["x_test"].shape[1] == NEW_SEQ_LEN

    # Check that the original part is unchanged.
    assert np.array_equal(original_x, marked_dataset["x"][:, :old_seq_len])
    assert np.array_equal(original_x_test, marked_dataset["x_test"][:, :old_seq_len])

    # Check dtype is preserved.
    assert marked_dataset["x"].dtype == original_x.dtype
    assert marked_dataset["x_test"].dtype == original_x_test.dtype

    # Check blank region.
    blank_start = old_seq_len
    blank_end = MARKER_POS

    assert np.all(
        marked_dataset["x"][:, blank_start:blank_end]
        == np.array(BLANK_VALUE, dtype=marked_dataset["x"].dtype)
    )

    assert np.all(
        marked_dataset["x_test"][:, blank_start:blank_end]
        == np.array(BLANK_VALUE, dtype=marked_dataset["x_test"].dtype)
    )

    # Check marker values.
    x = original_dataset["x"]
    x_test = original_dataset["x_test"]

    top_value = np.array(np.max(x), dtype=x.dtype)
    bottom_value = np.array(np.min(x), dtype=x.dtype)

    top_value_test = np.array(np.max(x_test), dtype=x_test.dtype)
    bottom_value_test = np.array(np.min(x_test), dtype=x_test.dtype)

    if not reversed_marker:
        assert np.all(marked_dataset["x"][y <= 4, MARKER_POS] == top_value)
        assert np.all(marked_dataset["x"][y >= 5, MARKER_POS] == bottom_value)

        assert np.all(marked_dataset["x_test"][y_test <= 4, MARKER_POS] == top_value_test)
        assert np.all(marked_dataset["x_test"][y_test >= 5, MARKER_POS] == bottom_value_test)

    else:
        assert np.all(marked_dataset["x"][y <= 4, MARKER_POS] == bottom_value)
        assert np.all(marked_dataset["x"][y >= 5, MARKER_POS] == top_value)

        assert np.all(marked_dataset["x_test"][y_test <= 4, MARKER_POS] == bottom_value_test)
        assert np.all(marked_dataset["x_test"][y_test >= 5, MARKER_POS] == top_value_test)


def main():
    with open(IN_PKL, "rb") as f:
        dataset = pickle.load(f)

    normal_dataset = add_group_marker(dataset, reversed_marker=False)
    reversed_dataset = add_group_marker(dataset, reversed_marker=True)

    check_marked_dataset(
        original_dataset=dataset,
        marked_dataset=normal_dataset,
        reversed_marker=False,
    )

    check_marked_dataset(
        original_dataset=dataset,
        marked_dataset=reversed_dataset,
        reversed_marker=True,
    )

    with open(OUT_PKL_NORMAL, "wb") as f:
        pickle.dump(normal_dataset, f, protocol=3)

    with open(OUT_PKL_REVERSED, "wb") as f:
        pickle.dump(reversed_dataset, f, protocol=3)

    print("Saved normal marked dataset to:")
    print(OUT_PKL_NORMAL)

    print()
    print("Saved reversed marked dataset to:")
    print(OUT_PKL_REVERSED)

    print()
    print("Original x shape:", dataset["x"].shape)
    print("Normal marked x shape:", normal_dataset["x"].shape)
    print("Reversed marked x shape:", reversed_dataset["x"].shape)

    print()
    print("Original x_test shape:", dataset["x_test"].shape)
    print("Normal marked x_test shape:", normal_dataset["x_test"].shape)
    print("Reversed marked x_test shape:", reversed_dataset["x_test"].shape)

    print()
    print("x dtype:", normal_dataset["x"].dtype)
    print("x_test dtype:", normal_dataset["x_test"].dtype)

    print()
    print("Marker position:", MARKER_POS)

    print()
    print("Normal marker:")
    print("Labels 0-4 marker value: top")
    print("Labels 5-9 marker value: bottom")

    print()
    print("Reversed marker:")
    print("Labels 0-4 marker value: bottom")
    print("Labels 5-9 marker value: top")

    plot_examples_with_group_marker(
        normal_dataset,
        title="Normal marker: labels 0-4 top value, labels 5-9 bottom value",
    )

    plot_examples_with_group_marker(
        reversed_dataset,
        title="Reversed marker: labels 0-4 bottom value, labels 5-9 top value",
    )


if __name__ == "__main__":
    main()