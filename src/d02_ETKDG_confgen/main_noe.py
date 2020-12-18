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

pdb_code = '2moa'
num_conf = 8
up_tol = 1

os.chdir('../../data/' + pdb_code.upper())
smiles_path = pdb_code.lower() + '.smi'
#smiles_path = '02_ref.smi'
ref_pdb_path = pdb_code.lower() + '.pdb'
#ref_pdb_path = '01_ref.pdb'
#ref_pdb_path = 'tmp.pdb'
xplor_path = pdb_code.lower() + '.mr'
#xplor_path = '10_DPep5_chcl3_noebounds.csv'

with open(smiles_path, "r") as tmp:
    smiles = tmp.read().replace("\n", "")
    smiles = smiles.replace('N+H2', 'NH2+')
    smiles = smiles.replace('N+H3', 'NH3+')

smiles_mol = Chem.MolFromSmiles(smiles)
#smiles_mol = Chem.RemoveAllHs(smiles_mol)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path, sanitize=True, removeHs=True)  # sanitizing done by assignment from smiles
ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol, ref_mol)


mol = assign_hydrogen_pdbinfo(Chem.AddHs(ref_mol))
#print(Chem.MolToPDBBlock(mol))

noe_df = parse_xplor(xplor_path)
noe_df = sanitize_xplor_noe(noe_df)
#noe_df = parse_csv(xplor_path, 'nm', sep="\s", comment="#")
#noe_df = sanitize_noe(noe_df)

prev_bmat = AllChem.GetMoleculeBoundsMatrix(mol, useMacrocycle14config=True)
bmat, rdmol_df = get_noe_restraint_bmat(mol, noe_df, up_tol=up_tol)

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

rdmol_df.to_csv(f'10_{pdb_code.lower()}_NOE_index.csv')

AllChem.EmbedMultipleConfs(mol, num_conf, params)
AllChem.AlignMolConformers(mol)  # align confs to first for easier visual comparison

Chem.MolToPDBFile(mol, "21_{}_tol{}_noebounds_{}.pdb".format(pdb_code.lower(), up_tol, num_conf))

runtime = time.time() - t
print(f"Took {runtime:.0f} s to generate {mol.GetNumConformers()} of demanded {num_conf} conformers.")
