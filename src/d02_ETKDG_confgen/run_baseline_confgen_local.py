import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os
import time
from src.d00_utils.utils import assign_hydrogen_pdbinfo, get_noe_restraint_bmat

"""
# give NOE file as first parameter, ref.pdb and ref.smi files in same directory
# number of conformers as second parameter
"""

t = time.time()

try:
    print("NOE file path: {}".format(sys.argv[1]))
except:
    print("Must specify path to NOE file.")
    exit()

try:
    num_conf = int(sys.argv[2])
    print("Number of conformers: {}".format(sys.argv[2]))
except:
    num_conf = 5400
    print("Defaulting number of conformers to {}".format(num_conf))

input_path = os.path.dirname(sys.argv[1])

print(input_path)

with open("02_ref.smi", "r") as tmp:
    smiles = tmp.read().replace("\n", "")

ref_pdb_path = "{}/01_ref.pdb".format(input_path)
ref_pdb_path = "01_ref.pdb"

smiles_mol = Chem.MolFromSmiles(smiles)
ref_mol = Chem.MolFromPDBFile(ref_pdb_path)
ref_mol = AllChem.AssignBondOrdersFromTemplate(smiles_mol,ref_mol)

    

mol = assign_hydrogen_pdbinfo(Chem.AddHs(ref_mol))
#print(Chem.MolToPDBBlock(mol))


params = AllChem.ETKDGv3()
params.useRandomCoords = False
params.randomSeed = 42
params.verbose = False
params.numThreads = 0           # use parallelism


AllChem.EmbedMultipleConfs(mol, num_conf, params)


Chem.MolToPDBFile(mol, "XX_BASELINE_{}_{}.pdb".format(sys.argv[1][:-4], num_conf))

runtime = time.time() - t
print(f"Took {runtime:.0f} s to generate {num_conf} conformers.")