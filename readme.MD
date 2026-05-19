# MNIST-1D Marker Dataset Workflow

This repository contains the MNIST-1D package, the base dataset, a script that creates
marker-augmented datasets, and the notebook used to run the classification experiments.

## 1. Clone the repository

The script and notebook currently use absolute paths for this project location:

```bash
/home/christian/FRMDL_toy_dataset/mnist1d
```

To use the repository without changing paths inside the script or notebook, clone it to
that location:

```bash
mkdir -p /home/christian/FRMDL_toy_dataset
cd /home/christian/FRMDL_toy_dataset
git clone git@github.com:cmplevier/toy_dataset_FRMDL.git mnist1d
cd mnist1d
```

If you clone somewhere else, update the absolute paths in:

- `mnist1d/SC_data_creation_run.py`
- `notebooks/mnist1dsc-classification_run.ipynb`

## 2. Create and activate the virtual environment

Create the virtual environment inside the repository:

```bash
python3 -m venv .venv_mnist1d
source .venv_mnist1d/bin/activate
```

Confirm that the active Python is the one from the virtual environment:

```bash
which python
```

Expected path:

```bash
/home/christian/FRMDL_toy_dataset/mnist1d/.venv_mnist1d/bin/python
```

## 3. Install the required libraries

With the virtual environment activated, install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Create the marker pickle files

From the repository root, run:

```bash
python mnist1d/SC_data_creation_run.py
```

This reads:

```bash
/home/christian/FRMDL_toy_dataset/mnist1d/mnist1d_data.pkl
```

and creates:

```bash
/home/christian/FRMDL_toy_dataset/mnist1d/mnist1d_data_x_length_50_with_marker.pkl
/home/christian/FRMDL_toy_dataset/mnist1d/mnist1d_data_x_length_50_with_marker_reversed.pkl
```

The generated datasets extend each MNIST-1D signal from length 40 to length 50. The
original signal is kept in positions `0-39`, positions `40-48` are blank, and the marker
is stored at position `49`.

The normal marker dataset uses:

- labels `0-4`: top marker value
- labels `5-9`: bottom marker value

The reversed marker dataset uses:

- labels `0-4`: bottom marker value
- labels `5-9`: top marker value

## 5. Register the virtual environment as a Jupyter kernel

Keep the virtual environment activated:

```bash
source .venv_mnist1d/bin/activate
```

Then register it as a Jupyter kernel:

```bash
python -m ipykernel install --user --name mnist1d-venv --display-name "Python (mnist1d venv)"
```

## 6. Run the notebook

Start Jupyter from the repository root:

```bash
jupyter notebook notebooks/mnist1dsc-classification_run.ipynb
```

In Jupyter, set the notebook kernel to:

```bash
Python (mnist1d venv)
```

If you are using VS Code, select this interpreter for the notebook:

```bash
/home/christian/FRMDL_toy_dataset/mnist1d/.venv_mnist1d/bin/python
```

The notebook expects the base dataset and both generated marker datasets to be present in
the repository root before running the experiment cells.

The notebook then produces the original mnist1d results and on top of that looks at the results of using MNIST1d-SC data to prove the properties that are discussed in the blog post. 
