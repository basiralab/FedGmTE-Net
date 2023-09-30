# FedGmTE-Net
FedGmTE-Net for predicting graph multi-trajectory evolution with the help of federated learning.

## Dependencies
Available to see on  requirements.txt file.

## Running FedGmTE-Net
We provide a demo code for the usage of FedGmTE-Net for predicting multi-trajectory evolution of graphs from a single baseline graph.
In main.py we train our network on a simulated dataset: 100 subjects in total with a 4-fold cross-validation. We use 25 of the subjects for testing and 75 of the subjects for training (we have 3 hospitals with 25 subjects each). Each subject has brain graphs denoting two modalities (i.e., low-resolution and super-resolution brain graphs) acquired at 3 timepoints. The first one is used as a baseline to train the model and predict the evolution trajectory including the rest of the timepoints as well. The user can modify all the program arguments and select the specific testing environment. Some of the possible user options are listed below:

* Modes (methods): NoFedGmTE-Net, FedGmTE-Net, FedGmTE-Net*
* Evaluation metrics: MAE(graph), MAE(NS)
* Data distributions: IID, non-IID (K-means split)
* Datasets: simulated dataset (100 subjects - 60% completed)

The user can also add hyper-parameters and vary their default values.

The user can run the code for training and testing with the following command:
```bash
python main.py
```

# Input and Output data
In order to use our framework, the user needs to provide a set of trajectories where a single trajectory represents a set of feature matrices acquired at multiple timepoints. A feature matrix is of size (n * d). We denote n the total number of subjects in the dataset and d the number of features extracted from the connectivity matrix. The user needs to make the appropriate changes in main.py and dataset.py to include their own data in a similar way as our simulated graphs. We note here that two feature matrices derived from two different trajecoties might have different number of features (i.e., super-resolution and low-resolution graphs). In that way, our code is generic and can be used for any type of isomorphic graph. In our example, for an input brain graph at t0 (35 x 35), our framework produces two trajectories each is a set of follow-up brain graphs  of a specific modality. The brain connectivity matrices of one modality have a size of 35 x 35 (morphological connectome) and for the second modality they have a size of 116 x 116 (functional connectome).
