import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from periodictable import elements

def initialize():
    global trimmed, unnested
    trimmed = []
    unnested = []
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
        x = Chem.AddHs(Chem.MolFromSmiles(x))
        AllChem.EmbedMolecule(x)
        vol = AllChem.ComputeMolVolume(x)
    except:
        vol = np.nan
    return vol

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
# each position on the amino acid sequence has its own column
data = pre_data['Sequence'].str.split(expand = True)
# one-hot encode protein sequence
data = pd.get_dummies(data).join(pre_data['SMILES'])
pre_data.drop(['SMILES'], axis = 1)
# get volumes of binding pockets
pre_data['Volume'] = data['SMILES'].apply(lambda x: compute_volume(x))
data = data.join(pre_data['Volume']).dropna()
Seq = list(data)[:1]
# rejoin protein sequence (temp)
data_Seq['Sequence'] = data[Seq].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
data = data_Seq.join(data['Volume'])
# renest dataframe
data = data.groupby('Sequence').agg(lambda x : ','.join(x))
# separate columns
data_Vol = data['Volume'].str.split(',', expand = True).add_prefix('Vol').fillna(value = 0, inplace = True)
data_Seq = data['Sequence'].str.split('-', expand = True).add_prefix('Seq')
data = data_Seq.join(Data_Vol)
random.shuffle(data)
# train and test dataframes
data_train = data.sample(frac = 0.8, random_state = 200) 
data_test = data.drop(data_train.index)
dataX_train = data.loc[:, ~data_train.columns.str.startswith('Vol')]
dataY_train = data.loc[:, ~data_train.columns.str.startswith('Seq')]
dataX_test = data.loc[:, ~data_test.columns.str.startswith('Vol')]
dataX_train.to_csv('dataX_train.csv', index = False)
dataY_train.to_csv('dataY_train.csv', index = False)
dataX_test.to_csv('dataX_test.csv', index = False)



