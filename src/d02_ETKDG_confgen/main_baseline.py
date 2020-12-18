import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
import sys
import os
from src.d00_utils.utils import *
import time
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count


t = time.time()

pdb_code = '6beu'
num_conf = 5400

os.chdir('../../data/30_Gene/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'
ref_pdb_path = pdb_code.lower() + '.pdb'

with open(smiles_path, "r") as tmp:
    smiles = tmp.read().replace("\n", "")
    smiles = smiles.replace('N+H2', 'NH2+')
    smiles = smiles.replace('N+H3', 'NH3+')

smiles_mol = Chem.MolFromSmiles(smiles)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path, sanitize=True, removeHs=True)  # sanitizing done by assignment from smiles

ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, ref_mol)
mol = assign_hydrogen_pdbinfo(Chem.AddHs(ref_mol))


bmat = AllChem.GetMoleculeBoundsMatrix(mol, useMacrocycle14config=True)

# check bmat
#comparison = bmat==prev_bmat
#identical = comparison.all()
#print(identical)

params = AllChem.ETKDGv3()
params.useRandomCoords = False #TODO changeable
params.SetBoundsMat(bmat)
params.verbose = False
params.randomSeed = 42
params.numThreads = 0
params.maxAttempts = 0
params.clearConfs = True

#pmol = mol
#todo_list = 10

#pool = Pool(processes=cpu_count())
#for i in tqdm(pool.imap_unordered(AllChem.EmbedMultipleConfs(mol, num_conf, params), todo_list), total=len(todo_list)):
#    print(i)

AllChem.EmbedMultipleConfs(mol, num_conf, params)


Chem.MolToPDBFile(mol, "20_{}_baseline_{}.pdb".format(pdb_code.lower(), num_conf))

runtime = time.time() - t
print(f"Took {runtime:.0f} s to generate {mol.GetNumConformers()} of demanded {num_conf} conformers.")
