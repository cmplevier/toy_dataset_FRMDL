# clone this library into your pc 
Git clone .... 

# create the relevant venv 
python3 -m venv .venv_mnist1d

# activate the venv and install all the libraries 
source .venv_mnist1d/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt


# presuming that you imported this library to your root, create the datasets with the markers 
# first create the normal marked
python3 ~/FRMDL_toy_dataset/mnist1d/mnist1d/SC_data_creation.py

#the comment out the first pcklout location and comment in the second one, also comment in the second indices.
then run it again. 

finally run the jupyter notebook called mnist1dsc-classification, while taking the virtual env as your python interpreter. 

