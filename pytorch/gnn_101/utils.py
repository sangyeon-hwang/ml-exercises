
from rdkit import Chem
import torch
import torch_geometric

ATOM_TYPES = ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]

def one_hot_encode(key, categories):
    """Raise ValueError if `key` is unknown."""
    code = torch.zeros(len(categories))
    code[categories.index(key)] = 1.
    return code

def get_atom_feature(atom):
    """atom: str | Chem.Atom"""
    if isinstance(atom, Chem.Atom):
        symbol = atom.GetSymbol()
    elif isinstance(atom, str):
        symbol = atom
    else:
        raise NotImplementedError
    feature = one_hot_encode(symbol, ATOM_TYPES)
    return feature

def get_bond_feature(bond):
    """bond: Chem.Bond | Chem.BondType"""
    if isinstance(bond, Chem.Bond):
        bond_type = bond.GetBondType()
    elif isinstance(bond, Chem.BondType):
        bond_type = bond
    else:
        raise NotImplementedError
    feature = one_hot_encode(bond_type, BOND_TYPES)
    return feature

def smiles_to_graph(smiles, label=None):
    """Return None if:
        1) the SMILES string is erroneous
        2) unknown atom or bond types exist
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    # Atom features & bond features
    try:
        # (num_atoms, num_features)
        node_features = torch.stack([get_atom_feature(atom)
                                     for atom in mol.GetAtoms()])
        # (num_bonds, num_features)
        edge_features = torch.stack([get_bond_feature(bond)
                                     for bond in mol.GetBonds()])
    # For unknown atom or bond types
    except ValueError:
        return

    # Edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        idx_pair = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        edge_indices.append(idx_pair)
        # Bidirectional
        edge_indices.append(idx_pair[::-1])
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)
    # (2, num_edges)
    edge_indices = edge_indices.t().contiguous()

    graph = torch_geometric.data.Data(x=node_features,
                                      edge_attr=edge_features,
                                      edge_index=edge_indices)

    # Attach the label if given.
    if label is not None:
        # Graph-level label: (1,)
        graph.y = torch.tensor(label, dtype=torch.float).unsqueeze(0)

    return graph
