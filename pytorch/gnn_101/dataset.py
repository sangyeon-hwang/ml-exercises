import os

import numpy as np
import pandas as pd
from rdkit import Chem
import torch_geometric

import utils

class MoleculeDataset(torch_geometric.data.Dataset):
    def __init__(self, csv_path):
        csv_path = os.path.realpath(os.path.expanduser(csv_path))
        dataframe = pd.read_csv(csv_path,
                                names=['smiles', 'label'],
                                dtype={'label': np.float32})
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        smiles, label = self.dataframe.iloc[idx]
        if np.isnan(label):
            graph = utils.smiles_to_graph(smiles)
        else:
            graph = utils.smiles_to_graph(smiles, label)
        return graph
