import os
import random

import numpy as np
import pandas as pd
from rdkit import Chem
import torch_geometric

import utils

class MoleculeDataset(torch_geometric.data.Dataset):
    def __init__(self, csv_path, num_data=None, max_num_atoms=None):
        csv_path = os.path.realpath(os.path.expanduser(csv_path))
        dataframe = pd.read_csv(csv_path,
                                names=['smiles', 'label'],
                                dtype={'label': np.float32})
        if num_data is not None:
            dataframe = dataframe.iloc[:num_data]
        self.dataframe = dataframe
        self.max_num_atoms = max_num_atoms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        smiles, label = self.dataframe.iloc[idx]
        if np.isnan(label):
            graph = utils.smiles_to_graph(smiles,
                                          max_num_atoms=self.max_num_atoms)
        else:
            graph = utils.smiles_to_graph(smiles, label, self.max_num_atoms)

        # `utils.smiles_to_graph` returns None
        # if error occurs during the graph creation.
        while graph is None:
            random.seed()
            idx = random.randint(0, len(self.dataframe))
            graph = self.__getitem__(idx)

        return graph
