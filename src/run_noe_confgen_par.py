import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os
from _utils import assign_hydrogen_pdbinfo, get_noe_restraint_bmat
import time
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

"""
!!!work in progress, not used!!!
give NOE file as first parameter, ref.pdb and ref.smi files in same directory
number of conformers as second parameter
"""

t = time.time()

try:
    print(f"NOE file path: {sys.argv[1]}")
except:
    print(f"Must specify path to NOE file.")
    exit()

try:
    num_conf = int(sys.argv[2])
    print(f"Number of conformers: {sys.argv[2]}")
except:
    num_conf = 5400
    print(f"Defaulting number of conformers to {num_conf}")
  
input_path = os.path.dirname(sys.argv[1])

#print(input_path)

with open("{}/ref.smi".format(input_path), "r") as tmp:
    smiles = tmp.read().replace("\n", "")

ref_pdb_path = "{}/ref.pdb".format(input_path)

smiles_mol = Chem.MolFromSmiles(smiles)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path)
ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)



mol = assign_hydrogen_pdbinfo(Chem.AddHs(ref_mol))
#print(Chem.MolToPDBBlock(mol))

# smiles
df = pd.read_csv(sys.argv[1], sep = "\s", comment = "#")

prev_bmat = AllChem.GetMoleculeBoundsMatrix(mol)
bmat = get_noe_restraint_bmat(mol, df)        

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

print(params.__dict__)

#pmol = mol
#todo_list = 10

#pool = Pool(processes=cpu_count())
#for i in tqdm(pool.imap_unordered(AllChem.EmbedMultipleConfs(mol, num_conf, params), todo_list), total=len(todo_list)):
#    print(i)

AllChem.EmbedMultipleConfs(mol, num_conf, params)


Chem.MolToPDBFile(mol, "{}_numconf{}.pdb".format(sys.argv[1][:-4], num_conf))

runtime = time.time() - t
print(f"Took {runtime:.0f} s to generate {num_conf} conformers.")