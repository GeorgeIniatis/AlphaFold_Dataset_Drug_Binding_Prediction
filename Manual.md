## Directory Structure

- `contactmaps/` contains the [nanoHUB tool](https://nanohub.org/resources/contactmaps) package that helped us calculate the protein contact maps
- `Dataset_Files/` contains various CSV files that were used to construct the dataset, these can be largely ignored as a pickle file of the dataset is provided
- `Dataset_Files/AlphaFold_Proteins` contains all the pdb files downloaded from [AlphaFold](https://alphafold.ebi.ac.uk/download)
- `Dataset_Files/Baseline_Models` & `Dataset_Files/Enhanced_Models` contain all our trained classification and regression models in the form of .joblib files
- `Dataset_Files/Feature_Selection` contains all the numpy files regarding the feature selection process 
- `Dataset_Files/Neural_Networks` contains the model checkpoint and its average train and validation losses
- `Dataset_Files/Protein_Graph_Data` cotains all the files needed to construct the protein graphs and the protein graphs themselves
- `Dataset_Files/Training_Test_Sets` contain the sets used in the training and testing phases for each model
- `Molecular_Functions_Embedding_Model_&_Files/` contains two notebooks. one that was used to create the dataset used by the embedding model, and the embedding model itself
- `Molecular_Functions_Embedding_Model_&_Files/Dataset_Files/` follows the same structure discussed 
- `Metrics/` contains all the metrics gathered from our trained models in the form of CSV files
- `R_Scripts/` contain the scriptrs that were used to calculate the amino acid and protein sequence descriptors
- `Dataset_Creation_&_Exploration.ipynb` was the Jupyter notebook used to bring together the various CSV files to create our dataset and split it into training and test sets
- `Classification_Baseline_Models.ipynb`,`Regression_Baseline_Models.ipynb`,`Classification_Enhanced_Models.ipynb` & `Regression_Enhanced_Models.ipynb` were the Jupyter Notebooks used to train and test our various models
- `DTIs_NN` was the Jypter notebook used to train and test the neural network for DTI prediction
-  `amino_acid_features.py`, `drug_features.py`, `protein_features.py`,`extract_dtis.py`, `models_utils.py` & `utils.py` contain helper functions that were used to create the various CSV files, the dataset and the models

## Requirements

* Python: 3.9.16
* PyTorch: 1.13.0
* PyTorch Geometric: 2.1.0
* Cuda Version: 1.17.0
* Packages: listed in `requirements.txt` 
* Tested on Windows 11

## Build steps

We would suggest the creation of an anaconda virtual environment and then running:

`pip install -r requirements.txt`

`conda install pytorch==1.13.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html`





