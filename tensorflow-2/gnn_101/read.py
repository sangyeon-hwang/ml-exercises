
from sklearn import preprocessing
import numpy as np
from rdkit import Chem
import tensorflow as tf

if __name__ == '__main__':
    table = Chem.rdchem.GetPeriodicTable()
    symbols = [table.GetElementSymbol(i) for i in range(1, 30)]

    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc.fit([[sym] for sym in symbols])
    print(enc.transform([['C'], ['C'], ['X']]).toarray())
