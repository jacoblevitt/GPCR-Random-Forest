import pandas as pd
import numpy as np
from keras.utils import to_categorical
from rdkit import Chem
from rdkit.Chem import MolSurf

alphabet = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

def onehot(x):
    try:
        tokens = list(x)
        vectors = [alphabet[i] for i in tokens]
        while len(vectors) <= maxlen:
            vectors.append(0)
        onehot = [to_categorical(i, num_classes = 21).tolist() for i in vectors]
        return onehot
    except:
        return np.nan

def compute_SASA(x):
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(str(x)))
        SASA = MolSurf.pyLabuteASA(mol)
        return SASA
    except:
        return np.nan

def unnest(x):
    try:
        unnested = []
        for i in x:
            flattened = []
            for j in i:
                for k in j:
                    flattened.append(k)
            unnested.append(flattened)
        return unnested
    except:
        pass
    
targets = pd.read_csv('targets.tsv', sep = '\t')[['FASTA Sequence', 'UniProt ID']]
ligands = pd.read_csv('ligands.tsv', sep = '\t')[['InChI Key', 'Canonical SMILES', 'Molecular Weight']]
ligands = ligands[ligands['Molecular Weight'].between(150, 500, inclusive = True)]
interactions = pd.read_csv('interactions_active.tsv', sep = '\t')[['UniProt ID', 'InChI Key']]
data = targets.merge(ligands.merge(interactions, on = 'InChI Key', how = 'outer'), on = 'UniProt ID', how = 'outer').dropna().drop_duplicates()[['FASTA Sequence', 'Canonical SMILES']]
maxlen = data['FASTA Sequence'].map(lambda x: len(x)).max()
data['Canonical SMILES'] = data['Canonical SMILES'].apply(lambda x: compute_SASA(x))
data = data.dropna()
data = data.groupby('FASTA Sequence', as_index = False).max()
data['FASTA Sequence'] = data['FASTA Sequence'].apply(lambda x: onehot(x))
data = data.dropna()
data_train = data.sample(frac = 0.8, random_state = 200)
data_test = data.drop(data_train.index)
dataX_train = np.array(unnest(data_train['FASTA Sequence'].tolist()))
dataY_train = np.array(data_train['Canonical SMILES'].tolist())
dataX_test = np.array(unnest(data_test['FASTA Sequence'].tolist()))
dataY_test = np.array(data_test['Canonical SMILES'].tolist())

np.savetxt('dataX_train.csv', dataX_train)
np.savetxt('dataY_train.csv', dataY_train)
np.savetxt('dataX_test.csv', dataX_test)
np.savetxt('dataY_test.csv', dataY_test)


    
