import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from periodictable import elements
from sklearn.ensemble import RandomForestRegressor

def initialize():
    global trimmed, unnested, vols
    trimmed = []
    unnested = []
    vols = {}
# remove parentheses from DrugBank IDs
def trim(x):
    if '>' in x:  
        start = x.find('(') 
        start += 1 
        end = x.find(')')  
        trimmed.append(x[start:end])   
    else:     
        trimmed.append(x)
# match DrugBank IDs to protein sequences
def unnest(x):
    for i in x:    
        if 'DB' in i:       
            try:  
                j = i.split('; ')  
                for k in j:      
                    unnested.append((k, str(x[x.index(i) + 1])))     
            except: 
                unnested.append((i, str(x[x.index(i) + 1])))

def organize(x):
    for i in x:     
        trim(i)    
    unnest(trimmed)
    return unnested
# remove drugs with unwanted atoms from the dataset
def subset(x, elements, sub):
    elim = [str(i) for i in elements if str(i) not in sub]
    for i in elim:
        x = x[~x.SMILES.str.contains(i)]
    return x

def compute_volume(x):
    try:
        x = str(x)
        if x in vols:
            vol = vols[x]
        else:
            mol = Chem.AddHs(Chem.MolFromSmiles(x))
            AllChem.EmbedMolecule(mol)
            vol = AllChem.ComputeMolVolume(mol)
            vol = str(np.round(vol, 3))
            vol_update = {x: vol}
            vols.update(vol_update)
    except:
        vol = np.nan
    print(vol)
    return vol

def replace_none(x):
    if x is None:
        x = str(0)
    return x

initialize()
# remove FASTA formatting
with open('protein.fasta', 'r') as f:    
    protein_data = f.read().replace('\n', '').replace(')', ')\n').replace('>', '\n>')    
    protein_data = protein_data.split('\n')
organize(protein_data)
protein_df = pd.DataFrame(unnested)
protein_df.columns = ['DrugBank ID', 'Sequence']
protein_df['Sequence'] = protein_df['Sequence'].apply(lambda x: x.replace('', ' '))
compound_df = pd.read_csv('structure links 5.csv')[['DrugBank ID', 'SMILES']]
# match drugs to protein sequences
pre_data = protein_df.merge(compound_df, on = 'DrugBank ID', how = 'outer').dropna().drop_duplicates()[['Sequence', 'SMILES']]
# limit atoms to sub
sub = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Se', 'I'] 
pre_data = subset(pre_data, elements, sub)
pre_data = pre_data
# each position on the amino acid sequence has its own column
data = pre_data['Sequence'].str.split(expand = True).fillna(value = '0')
# one-hot encode protein sequence
data = pd.get_dummies(data) 
features = list(data)[:-1]
# get volumes of binding pockets
pre_data['Volume'] = pre_data['SMILES'].apply(lambda x: compute_volume(x))
data = data.join(pre_data['Volume']).dropna()
# rejoin protein sequence (temp)
data_seq = data[features].apply(lambda x: '-'.join(x.values.astype(str)), axis = 1)
data_seq = data_seq.to_frame()
data = data_seq.join(data['Volume'])
data.columns = ['Sequence', 'Volume']
# renest dataframe
data = data.groupby('Sequence')['Volume'].agg(','.join).reset_index(name = 'Volume')
# separate columns
data_vol = data['Volume'].str.split(',', expand = True).add_prefix('Vol').fillna(value = '0')
data_seq = data['Sequence'].str.split('-', expand = True).add_prefix('Seq')
data_seq.columns = [features]
data = data_seq.join(data_vol)
print(data)
# train and test dataframes
data_train = data.sample(frac = 0.8, random_state = 200) 
data_test = data.drop(data_train.index)
dataX_train = data_train[data_train.columns[-len(list(data_vol)):]]
dataY_train = data_train[data_train.columns[len(features):]]
dataX_test = data_test[data_test.columns[-len(list(data_vol)):]]
dataY_test = data_test[data_test.columns[len(features):]]
dataX_train.to_csv('dataX_train.csv', index = False)
dataY_train.to_csv('dataY_train.csv', index = False)
dataX_test.to_csv('dataX_test.csv', index = False)
dataY_test.to_csv('dataY_test.csv', index = False)

