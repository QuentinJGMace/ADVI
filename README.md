# Introduction
This repository is the code corresponding to our project on ADVI (Automatic Differentiation Variational Inference) optimization method made for the MVA class : graphical models, discrete inference and learning

# How to reproduce experiments
- download the porto taxi trajectories dataset available <a href="https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015">here</a>
- replace FILE_NAME in porto_preprocess.py by the train.csv file path of the taxi dataset
- execute porto_preprocess.py (it can take a while as it goes through the entire dataset)
- In the taxi.ipynb notebook replace array_path by the path of the array created by porto_preprocess.py
- You can go through the notebook (we advise not to execute the notebook as it takes quite a long time to run)

# Repository's structure

- advi.py : File containing the ADVI optimization object
- ppca.py : File containing probabilistic PCA implementation (actually there are 3 different versions, not all equivalent)
- gmm.py : File containing Gaussian Mixture model implementation
- porto_preprocess.py : File for preprocessing the taxi dataset

- toy_data.ipynb : Notebook for our experimentations on simple 2D data
- taxis.ipynb : Notebook for our experiment on the taxi_dataset

# Contributors
- Gabriel Ben Zenou
- Quentin Macé
- Alexandre Selvestrel
