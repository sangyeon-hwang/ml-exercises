import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

def read_data(path, num_data=None):
    molecule_list = []
    label_list = []
    with open(path) as f:
        for i, line in enumerate(f):
            smiles, label = line.strip().split(',')
            #molecule_list.append(Chem.AddHs(Chem.MolFromSmiles(smiles)))
            molecule_list.append(Chem.MolFromSmiles(smiles))
            label_list.append(label)  # List[str]
            if num_data and i+1 >= num_data:
                break
    return molecule_list, label_list

def process_molecule(molecule, max_num_atoms):
    num_atoms = molecule.GetNumAtoms()
    assert num_atoms <= max_num_atoms

    # Atom features
    features = []
    for i in range(num_atoms):
        atomic_number = molecule.GetAtomWithIdx(i).GetAtomicNum()
        if atomic_number > 1:
            features.append([0, 1])
        else:
            features.append([1, 0])
    #features = [[atom.GetAtomicNum()] for atom in molecule.GetAtoms()]

    # Adjacency matrix
    adj = GetAdjacencyMatrix(molecule)

    # Padding
    diff = max_num_atoms - num_atoms
    padded_features = np.pad(features, ((0, diff), (0, 0)))
    padded_adj = np.pad(adj, (0, diff))
    return padded_adj, padded_features

def load_data(path, max_num_atoms, num_data=None):
    molecule_list, label_list = read_data(path, num_data)
    adj_list, feature_list = zip(*[process_molecule(molecule, max_num_atoms)
                                   for molecule in molecule_list
                                   if molecule.GetNumAtoms() <= max_num_atoms])

    # Set the data type.
    adj_list = np.array(adj_list, dtype='float32')
    feature_list = np.array(feature_list, dtype='float32')
    label_list = np.array(label_list, dtype='float32')
    return adj_list, feature_list, label_list
